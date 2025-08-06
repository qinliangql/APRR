export CUDA_VISIBLE_DEVICES=0
CUDA_LAUNCH_BLOCKING=0


python projects/habitat_ovmm/eval_baselines_agent.py \
    --env_config projects/habitat_ovmm/configs/env/hssd_demo.yaml \
    --evaluation_type local \
    --agent_type play_rl \
    --baseline_config_path projects/habitat_ovmm/configs/agent/heuristic_agent_rl.yaml\
    habitat.dataset.split=val habitat.dataset.episode_indices_range=[100,150] 

