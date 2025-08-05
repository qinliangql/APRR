# export AGENT_EVALUATION_TYPE=local
# export LOCAL_ARGS=habitat.dataset.split=minival

# python projects/habitat_ovmm/eval_baselines_agent.py --env_config projects/habitat_ovmm/configs/env/hssd_demo.yaml

export DISPLAY=192.168.161.236:0.0
export CUDA_VISIBLE_DEVICES=1
CUDA_LAUNCH_BLOCKING=0
export PYTHONPATH=/aiarena/gpfs/code/code/OVMM/home-robot:$PYTHONPATH


python projects/habitat_ovmm/eval_baselines_agent.py \
    --env_config projects/habitat_ovmm/configs/env/hssd_base.yaml \
    --evaluation_type local \
    --agent_type base_s1 \
    --baseline_config_path projects/habitat_ovmm/configs/agent/heuristic_agent.yaml\
    habitat.dataset.split=val habitat.dataset.episode_indices_range=[0,10] 

# python projects/habitat_ovmm/eval_baselines_agent.py \
#     --env_config projects/habitat_ovmm/configs/env/hssd_demo.yaml \
#     --evaluation_type local \
#     --agent_type play \
#     --baseline_config_path projects/habitat_ovmm/configs/agent/heuristic_agent_play.yaml\
#     habitat.dataset.split=minival habitat.dataset.episode_indices_range=[0,10] 

# for ((i = 0; i < 500; i++)); do
#     python projects/habitat_ovmm/eval_baselines_agent.py \
#         --env_config projects/habitat_ovmm/configs/env/hssd_demo.yaml \
#         --evaluation_type local \
#         --agent_type play \
#         --baseline_config_path projects/habitat_ovmm/configs/agent/heuristic_agent_play.yaml\
#         habitat.dataset.split=minival habitat.dataset.episode_indices_range=[0,10] 
#     # python projects/habitat_ovmm/eval_baselines_agent.py \
#     #     --env_config projects/habitat_ovmm/configs/env/hssd_demo.yaml \
#     #     --evaluation_type local \
#     #     --agent_type play \
#     #     --baseline_config_path projects/habitat_ovmm/configs/agent/heuristic_agent_play.yaml\
#     #     habitat.dataset.split=val habitat.dataset.episode_indices_range=[0,5] 
# done

# python projects/habitat_ovmm/eval_baselines_agent.py --evaluation_type local habitat.dataset.split=minival

# python projects/habitat_ovmm/eval_baselines_agent.py --env_config projects/habitat_ovmm/configs/env/hssd_demo.yaml


# wget https://huggingface.co/datasets/hssd/hssd-hab/resolve/ovmm/objects/d/d26ce04e8e3f3dce915387fd82d1703a6d5d5b99.glb \
#     -P /aiarena/gpfs/code/code/OVMM/home-robot/data/cache

# ln -s /aiarena/group/eaigroup/ql/ovmm/train_data /aiarena/gpfs/code/code/OVMM/home-robot/

# ln -s /data/qinl/data_share/home-robot/datadump /aiarena/gpfs/code/code/OVMM/home-robot/
# ln -s /aiarena/group/eaigroup/ql/ovmm/datadump /aiarena/gpfs/code/code/OVMM/home-robot/
# ln -s /aiarena/group/eaigroup/ql/TEST/mytest /aiarena/gpfs/code/code/OVMM/home-robot/

#!/bin/bash

# # 目标文件夹
# target_folder="/aiarena/gpfs/code/code/OVMM/home-robot/data/hssd-hab/objects/e/"
# # 下载文件的基URL
# base_url="https://huggingface.co/datasets/hssd/hssd-hab/resolve/ovmm/objects/e"

# # 遍历目标文件夹，查找所有 .glb 文件
# for file in "$target_folder"/*.glb
# do
#     # 获取文件名（不包括路径）
#     filename=$(basename "$file")
    
#     # 构造完整的下载URL
#     url="$base_url/$filename"
    
#     # 下载文件到指定文件夹
#     wget -P "/aiarena/gpfs/code/code/OVMM/home-robot/data/cache/e" "$url"
    
#     # 打印下载状态
#     if [ $? -eq 0 ]; then
#         echo "Downloaded $filename successfully."
#     else
#         echo "Failed to download $filename."
#     fi
# done
