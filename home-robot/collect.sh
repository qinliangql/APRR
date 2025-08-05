export CUDA_VISIBLE_DEVICES=6

python projects/habitat_ovmm/eval_baselines_agent.py \
    --env_config projects/habitat_ovmm/configs/env/hssd_collect.yaml \
    --evaluation_type local \
    habitat.dataset.split=minival \
    habitat.dataset.episode_indices_range=[5,10]