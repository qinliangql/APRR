import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from einops import rearrange, repeat
# from transformers import CLIPModel, CLIPProcessor
from torch.nn.parallel import DistributedDataParallel as DDP
import math

class PatchEmbed(nn.Module):
    """将2D特征图转换为patches并进行embedding"""
    def __init__(self, patch_size=8, in_channels=128, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # 输入: [B, C=64, H=240, W=240]
        B, C, H, W = x.shape
        # Conv2d降采样: [B, embed_dim=128, H//patch_size=15, W//patch_size=15]
        x = self.proj(x)
        # 重排序: [B, 225, 128], 其中225 = 15 * 15
        x = rearrange(x, 'b c h w -> b (h w) c')
        # LayerNorm: [B, 225, 128]
        x = self.norm(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 扩展维度: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # 输入: [B, 225, 128]
        # 输出: [B, 225, 128]
        return x + self.pe[:, :x.size(1)]

class GetLocalFeature(nn.Module):
    """Enhanced attention module that explicitly handles both local and global context"""
    def __init__(self, dim=256, local_size=120):
        super().__init__()
        self.local_size = local_size
        self.dim = dim
        
    def get_agent_pos(self, local_map):
        """从local_map中提取agent位置"""
        current_pos = local_map[:, 1]  # agent position channel
        B = current_pos.shape[0]
        pos_y, pos_x = [], []
        
        for b in range(B):
            y, x = torch.where(current_pos[b] == 1)
            if len(y) > 0:
                # 计算所有agent位置的平均值
                pos_y.append(y.float().mean().round().long())
                pos_x.append(x.float().mean().round().long())
            else:
                # 如果没有找到agent，使用地图中心作为默认位置
                pos_y.append(torch.tensor(current_pos.shape[1]//2))
                pos_x.append(torch.tensor(current_pos.shape[2]//2))
        
        return torch.stack([torch.tensor(pos_y), torch.tensor(pos_x)], dim=1).to(local_map.device)
    
    def extract_local_region(self, local_map, agent_pos):
        """直接从local_map提取局部区域"""
        B, C, H, W = local_map.shape
        device = local_map.device
        
        half_size = self.local_size // 2
        y_pos = agent_pos[:, 0]  
        x_pos = agent_pos[:, 1]
        
        # 直接裁剪局部区域
        local_regions = []
        for b in range(B):
            y, x = y_pos[b], x_pos[b]
            # 确保不越界
            y_start = max(0, y - half_size)
            y_end = min(H, y + half_size)
            x_start = max(0, x - half_size)
            x_end = min(W, x + half_size)
            
            region = local_map[b:b+1, :, y_start:y_end, x_start:x_end]
            # 如果需要填充到固定大小
            if region.shape[2:] != (self.local_size, self.local_size):
                region = F.interpolate(
                    region, 
                    size=(self.local_size, self.local_size),
                    mode='bilinear',
                    align_corners=False
                )
            local_regions.append(region)
            
        return torch.cat(local_regions, dim=0)
    
    def forward(self, local_map):
        # Get agent position
        agent_pos = self.get_agent_pos(local_map)

        # 从原始local_map提取局部区域
        local_region = self.extract_local_region(local_map, agent_pos)
        
        return local_region

class SpatialTokenizer(nn.Module):
    """将空间特征转换为tokens"""
    def __init__(self, in_channels, patch_size, embed_dim, max_len=512):
        super().__init__()
        # Patch embedding: [B, C, H, W] -> [B, H//patch_size*W//patch_size, embed_dim]
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        
        # 位置编码: [B, 225, 128]
        self.pos_embed = PositionalEncoding(embed_dim,max_len=max_len)
        
    def forward(self, x):
        # 2. Patch embedding
        x = self.patch_embed(x)  # [B, 225, embed_dim]
        # 3. 添加位置编码
        x = self.pos_embed(x)  # [B, 225, embed_dim]
        return x

class SpatialEncoder(nn.Module):
    """编码空间信息(distance map和agent map)的模块"""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # 处理120x120的输入
            nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1),  # -> 60x60
            nn.LayerNorm([64, 60, 60]),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> 30x30
            nn.LayerNorm([128, 30, 30]),
            nn.GELU(),
            nn.Conv2d(128, hidden_dim, kernel_size=3, stride=2, padding=1),  # -> 15x15
            nn.LayerNorm([hidden_dim, 15, 15]),
            nn.GELU()
        )
        
        self.tokenizer = SpatialTokenizer(in_channels=hidden_dim, patch_size=1, embed_dim=hidden_dim,max_len=225)
        
        self.agent_local_process = GetLocalFeature()
        
    def forward(self, local_map):
        x = self.agent_local_process(local_map)
        # Process through conv layers: [B, hidden_dim, 15, 15]
        x = self.conv_layers(x)
        
        # Convert to sequence of patches: [B, 225, hidden_dim]
        x = self.tokenizer(x)
        
        return x

class StateProcessor(nn.Module):
    """处理状态向量信息(goal_x - x, goal_y - y, angle)"""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.state_embed = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.goal_iou_tokenizer = SpatialTokenizer(in_channels=1, patch_size=160, embed_dim=256,max_len=12)
        self.fc = nn.Linear(12*256, 12)
        
        # 合并编码器
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + 12, hidden_dim),
        )
        
    def forward(self, agent_state, goal_iou):
        agent_feat = self.state_embed(agent_state)
        
        goal_iou_feat = self.goal_iou_tokenizer(goal_iou)
        goal_iou_feat = goal_iou_feat.view(agent_feat.size(0), -1)
        goal_iou_feat = self.fc(goal_iou_feat)  # 将最后一个维度展平
        
        # 融合特征
        combined_feat = self.fusion_layer(
            torch.cat([agent_feat, goal_iou_feat], dim=-1)
        )
        return combined_feat

class MemoryTransformer(nn.Module):
    """处理历史信息的Transformer"""
    def __init__(self, hidden_dim=256, nhead=8, num_layers=1):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                dropout=0.01,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # 用于生成初始隐状态
        self.init_memory = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # 状态更新门
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, spatial_features, state_features, prev_memory=None):
        B = spatial_features.size(0)
        
        if prev_memory is None:
            prev_memory = self.init_memory.expand(B, -1, -1)
            
        # 将spatial features和state features结合
        # spatial_features: [B, 225, hidden_dim]
        # state_features: [B, hidden_dim] -> [B, 1, hidden_dim]
        state_features = state_features.unsqueeze(1)
        
        # 合并所有特征: [B, 226, hidden_dim]
        combined_features = torch.cat([spatial_features, state_features], dim=1)
        
        # 通过transformer处理: [B, 226, hidden_dim]
        transformed = self.transformer(combined_features)
        
        # 提取新的记忆状态（使用最后一个token）
        new_memory_candidate = transformed[:, -1:, :]
        
        # 计算更新门值
        gate = self.update_gate(
            torch.cat([prev_memory, new_memory_candidate], dim=-1)
        )
        
        # 更新记忆
        new_memory = gate * new_memory_candidate + (1 - gate) * prev_memory
        
        return transformed, new_memory

def init_type_embeddings(hidden_dim, num_types=5):
    position = torch.arange(num_types).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
    embeddings = torch.zeros(num_types, hidden_dim)
    embeddings[:, 0::2] = torch.sin(position * div_term)
    embeddings[:, 1::2] = torch.cos(position * div_term)
    return nn.Parameter(embeddings)

class ActionDecoder(nn.Module):
    """解码动作的模块"""
    def __init__(self, hidden_dim=256, num_actions=5):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 5, hidden_dim))
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=8, 
            batch_first=True
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(5*hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
    def forward(self, features):
        B = features.size(0)
        query = self.query.expand(B, -1, -1)
        
        # Cross attention
        attn_out, _ = self.attention(query, features, features)
        attn_out = attn_out.reshape(B, -1) 
        # Predict actions
        actions = self.action_head(attn_out.squeeze(1))
        return actions

class NavigatorWithAction(nn.Module):
    def __init__(self, hidden_dim=256, num_actions=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 编码器
        self.spatial_encoder = SpatialEncoder(hidden_dim)
        self.state_processor = StateProcessor(hidden_dim)
        
        # 记忆处理器
        self.memory_transformer = MemoryTransformer(hidden_dim)
        
        # 动作解码器
        self.action_decoder = ActionDecoder(hidden_dim, num_actions)
        
    def forward(self, local_map, agent_state, goal_iou, memory=None):
        # 1. 编码空间信息
        spatial_features = self.spatial_encoder(local_map)
        
        # 2. 处理状态向量
        state_features = self.state_processor(agent_state, goal_iou)
        
        # 3. 通过memory transformer处理
        transformed_features, new_memory = self.memory_transformer(
            spatial_features, 
            state_features, 
            memory
        )
        
        # 4. 解码动作
        actions = self.action_decoder(transformed_features)
        
        return actions, new_memory

# 初始化模型
def initialize_model(rank):
    model = NavigatorWithAction()
    model = model.to(rank)
    model = DDP(model, device_ids=[rank],find_unused_parameters=True)
    return model

def DistanceAwareBCELoss(inputs, targets, weighted_map):
    # 标准二元交叉熵
    BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
    
    # 基于预测图与目标的差异计算惩罚图
    penalty_map = 1.0 - weighted_map  # 目标的反转，使远离目标的区域有更高的惩罚
    
    # 用这个惩罚图缩放 BCE 损失
    weighted_loss = BCE_loss * (1.0 + 2*penalty_map)  # 为远离目标的区域添加惩罚
    return torch.mean(weighted_loss) * 1000

class FocalLoss(nn.Module):
    def __init__(self,device, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.tensor([0.35,0.10,0.12,0.12,0.35]).to(device)
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
            
        return focal_loss.mean()*10