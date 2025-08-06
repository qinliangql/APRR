export CUDA_VISIBLE_DEVICES=1
CUDA_LAUNCH_BLOCKING=0


python projects/habitat_ovmm/eval_baselines_agent.py \
    --env_config projects/habitat_ovmm/configs/env/hssd_base.yaml \
    --evaluation_type local \
    --agent_type base_s1 \
    --baseline_config_path projects/habitat_ovmm/configs/agent/heuristic_agent.yaml\
    habitat.dataset.split=val habitat.dataset.episode_indices_range=[100,150] 
