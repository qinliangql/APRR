#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple
from collections import OrderedDict
from habitat.core.agent import Agent
from habitat.core.spaces import EmptySpace
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.common import batch_obs
import gym.spaces as spaces
import home_robot.utils.pose as pu
from home_robot.core.interfaces import DiscreteNavigationAction, Observations

class NavResNetEncoder(nn.Module):
    """
    Simple CNN encoder based on ResNet architecture
    """
    def __init__(self, observation_space):
        super().__init__()
        # Define image size after transforms (from config)
        self.height = 160  # Comes from center_cropper height
        self.width = 120   # Comes from center_cropper width
        
        # Define network layers
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # For depth + 2 segmentation channels
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate flattened size
        self._flattened_size = self._get_conv_output_size()
        
    def _get_conv_output_size(self):
        """Helper function to calculate the size of CNN output"""
        dummy_input = torch.zeros(1, 3, self.height, self.width)
        return self.visual_encoder(dummy_input).shape[1]
    
    def forward(self, observations):
        # Get visual observations (depth + segmentations)
        visual_obs = []
        
        if "robot_head_depth" in observations:
            visual_obs.append(observations["robot_head_depth"])
            
        if "ovmm_nav_goal_segmentation" in observations:
            visual_obs.append(observations["ovmm_nav_goal_segmentation"])
            
        if "receptacle_segmentation" in observations:
            visual_obs.append(observations["receptacle_segmentation"])
            
        visual_obs = torch.cat(visual_obs, dim=1)  # Concatenate along channel dimension
        
        # Process through CNN
        visual_features = self.visual_encoder(visual_obs)
        return visual_features

class NavPolicy(nn.Module):
    """
    Policy network for navigation with discrete actions
    """
    def __init__(self, observation_space, hidden_size=512, num_recurrent_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_recurrent_layers = num_recurrent_layers
        
        self.net = NavResNetEncoder(observation_space)
        
        # LSTM layers
        self.state_encoder = nn.LSTM(
            self.net._flattened_size,
            hidden_size,
            num_recurrent_layers,
            batch_first=True,
        )
        
        # Define number of actions (move_forward, turn_left, turn_right, stop)
        self.num_actions = 4
        
        # Policy head (actor)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )
        
        # Value head (critic)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        visual_features = self.net(observations)
        
        # Reshape for LSTM
        N = visual_features.size(0)
        visual_features = visual_features.view(N, 1, -1)
        
        # LSTM forward pass
        rnn_features, rnn_hidden_states = self.state_encoder(
            visual_features,
            (
                rnn_hidden_states[0].contiguous(),
                rnn_hidden_states[1].contiguous(),
            ),
        )
        
        # Get features from last LSTM layer
        features = rnn_features[:, -1]
        
        # Get action distribution and value
        action_logits = self.action_head(features)
        value = self.critic(features)
        
        return action_logits, value, rnn_hidden_states

class NavPPOPolicy(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size=512,
        num_recurrent_layers=2,
        **kwargs
    ):
        super().__init__()
        self.net = NavPolicy(
            observation_space,
            hidden_size=hidden_size,
            num_recurrent_layers=num_recurrent_layers,
        )
        
        self.dim_actions = action_space.n
        
        # Define action distribution
        self.action_distribution = torch.distributions.Categorical
        
        self.critic_linear = self.net.critic
        self.num_recurrent_layers = num_recurrent_layers
        
    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)
        
    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        action_logits, value, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        
        dist = self.action_distribution(logits=action_logits)
        
        if deterministic:
            actions = dist.mode()
        else:
            actions = dist.sample()
            
        action_log_probs = dist.log_prob(actions)
        
        return value, actions, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        _, value, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        return value

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        action_logits, value, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        dist = self.action_distribution(logits=action_logits)
        
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return value, action_log_probs, dist_entropy, rnn_hidden_states

class PPOAgent(Agent):
    def __init__(
        self,
        config,
        skill_config,
        device_id: int = 0,
        obs_spaces=None,
        action_spaces=None,
    ) -> None:
        # Initialize basic parameters
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.config = config
        
        # Set observation and action spaces
        if obs_spaces is None:
            # Define your observation space here based on what you need
            self.obs_space = spaces.Dict({
                "rgb": spaces.Box(0, 1, (160, 120, 3), dtype=np.float32),
                "depth": spaces.Box(0, 1, (160, 120, 1), dtype=np.float32),
                "local_map": spaces.Box(0, 1, (240, 240, 5), dtype=np.float32),
            })
        else:
            self.obs_space = obs_spaces[0]
            
        # Define discrete action space
        if action_spaces is None:
            self.action_space = spaces.Discrete(5)  # move_forward, turn_left, turn_right, stop, move_forward_small
        else:
            self.action_space = action_spaces[0]
            
        # Initialize policy network
        self.actor_critic = NavPPOPolicy(
            observation_space=self.obs_space,
            action_space=self.action_space,
            hidden_size=512,
            num_recurrent_layers=2,
        ).to(self.device)
        
        # Load checkpoint if provided
        if hasattr(skill_config, 'checkpoint_path') and skill_config.checkpoint_path:
            ckpt = torch.load(skill_config.checkpoint_path, map_location=self.device)
            self.actor_critic.load_state_dict(
                {k[len("actor_critic."):]: v for k, v in ckpt["state_dict"].items() if "actor_critic" in k}
            )
            
        self.actor_critic.eval()
        
        # Navigation parameters
        self.discrete_forward = skill_config.discrete_forward
        self.discrete_turn = skill_config.discrete_turn_degrees * np.pi / 180
        
        # Initialize agent state
        self.reset()
    
    def reset(self) -> None:
        """Reset agent state at the start of a new episode."""
        self.test_recurrent_hidden_states = torch.zeros(
            1,
            self.actor_critic.num_recurrent_layers,
            512,  # hidden_size
            device=self.device,
            dtype=torch.float32,
        )
        self.not_done_masks = torch.zeros(1, 1, device=self.device, dtype=torch.bool)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        
    def act(self, observations: Observations) -> Tuple[DiscreteNavigationAction, dict, bool]:
        """Process observations and return navigation action."""
        # Convert observations to correct format and device
        batch = batch_obs([observations], device=self.device)
        
        with torch.no_grad():
            value, action, action_log_prob, self.test_recurrent_hidden_states = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )
            
            # Update previous action
            self.prev_actions.copy_(action)
            # Update masks
            self.not_done_masks.fill_(True)
            
            # Convert action to navigation command
            discrete_action = action.item()
            robot_action = self._convert_action_to_nav(discrete_action)
            
            # Check if agent wants to stop
            does_want_terminate = (discrete_action == 3)  # 3 is the STOP action
            
            return robot_action, {}, does_want_terminate
            
    def _convert_action_to_nav(self, discrete_action: int) -> DiscreteNavigationAction:
        """Convert network output to navigation action."""
        if discrete_action == 0:  # FORWARD
            return ContinuousNavigationAction(np.array([self.discrete_forward, 0.0, 0.0]))
        elif discrete_action == 1:  # TURN LEFT
            return ContinuousNavigationAction(np.array([0.0, 0.0, self.discrete_turn]))
        elif discrete_action == 2:  # TURN RIGHT
            return ContinuousNavigationAction(np.array([0.0, 0.0, -self.discrete_turn]))
        elif discrete_action == 3:  # STOP
            return DiscreteNavigationAction.STOP
        else:
            raise ValueError(f"Unknown discrete action: {discrete_action}")