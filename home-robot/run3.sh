# export AGENT_EVALUATION_TYPE=local
# export LOCAL_ARGS=habitat.dataset.split=minival

# python projects/habitat_ovmm/eval_baselines_agent.py --env_config projects/habitat_ovmm/configs/env/hssd_demo.yaml

export CUDA_VISIBLE_DEVICES=1
export DISPLAY=192.168.161.236:0.0
python projects/habitat_ovmm/eval_baselines_agent.py \
    --env_config projects/habitat_ovmm/configs/env/hssd_eval.yaml \
    --agent_type play \
    --baseline_config_path projects/habitat_ovmm/configs/agent/heuristic_agent_play.yaml\
    --evaluation_type local habitat.dataset.split=minival habitat.dataset.episode_indices_range=[0,5]
# python projects/habitat_ovmm/eval_baselines_agent.py --evaluation_type local habitat.dataset.split=val
# python projects/habitat_ovmm/eval_baselines_agent.py --env_config projects/habitat_ovmm/configs/env/hssd_demo.yaml

# wget https://huggingface.co/datasets/hssd/hssd-hab/resolve/ovmm/objects/e/ebe95e90df6cefd0e3af6dbc320b82323535a540.glb -P /aiarena/gpfs/code/code/OVMM/home-robot/data/cache

# random agent
# pybullet build time: May 20 2022 19:45:31
# 2024-09-13 09:12:35,884 Initializing dataset OVMMDataset-v0
# 2024-09-13 09:12:36,516 initializing sim OVMMSim-v0
# [09:12:38:592576]:[Core] ManagedContainerBase.h(329)::checkExistsWithMessage : <Lighting Layout>::getObjectByHandle : Unknown Lighting Layout managed object handle :  . Aborting
# 2024-09-13 09:12:39,515 Initializing task OVMMNavToObjTask-v0
# /aiarena/gpfs/miniconda3/envs/home-robot/lib/python3.9/site-packages/gym/spaces/box.py:84: UserWarning: WARN: Box bound precision lowered by casting to float32
#   logger.warn(f"Box bound precision lowered by casting to {self.dtype}")
#   6%|████████▏                                                                                                                         | 75/1199 [03:40<18:39,  1.00it/s][09:16:20:903644]:[Core] ManagedContainerBase.h(329)::checkExistsWithMessage : <Lighting Layout>::getObjectByHandle : Unknown Lighting Layout managed object handle :  . Aborting
#  15%|██████████████████▍                                                                                                            | 174/1199 [07:27<1:33:23,  5.47s/it][09:20:07:806180]:[Core] ManagedContainerBase.h(329)::checkExistsWithMessage : <Lighting Layout>::getObjectByHandle : Unknown Lighting Layout managed object handle :  . Aborting
#  23%|█████████████████████████████                                                                                                  | 274/1199 [12:50<1:13:21,  4.76s/it][09:25:31:571003]:[Core] ManagedContainerBase.h(329)::checkExistsWithMessage : <Lighting Layout>::getObjectByHandle : Unknown Lighting Layout managed object handle :  . Aborting
#  31%|████████████████████████████████████████▏                                                                                        | 374/1199 [17:28<40:07,  2.92s/it][09:30:08:873286]:[Core] ManagedContainerBase.h(329)::checkExistsWithMessage : <Lighting Layout>::getObjectByHandle : Unknown Lighting Layout managed object handle :  . Aborting
#  40%|██████████████████████████████████████████████████▉                                                                              | 474/1199 [22:18<40:59,  3.39s/it][09:34:58:998069]:[Core] ManagedContainerBase.h(329)::checkExistsWithMessage : <Lighting Layout>::getObjectByHandle : Unknown Lighting Layout managed object handle :  . Aborting
#  48%|█████████████████████████████████████████████████████████████▊                                                                   | 574/1199 [27:55<27:10,  2.61s/it][09:40:37:641916]:[Core] ManagedContainerBase.h(329)::checkExistsWithMessage : <Lighting Layout>::getObjectByHandle : Unknown Lighting Layout managed object handle :  . Aborting
#  56%|████████████████████████████████████████████████████████████████████████▌                                                        | 674/1199 [33:57<12:16,  1.40s/it][09:46:38:534559]:[Core] ManagedContainerBase.h(329)::checkExistsWithMessage : <Lighting Layout>::getObjectByHandle : Unknown Lighting Layout managed object handle :  . Aborting
#  65%|███████████████████████████████████████████████████████████████████████████████████▎                                             | 774/1199 [39:51<20:42,  2.92s/it][09:52:32:166567]:[Core] ManagedContainerBase.h(329)::checkExistsWithMessage : <Lighting Layout>::getObjectByHandle : Unknown Lighting Layout managed object handle :  . Aborting
#  73%|██████████████████████████████████████████████████████████████████████████████████████████████                                   | 874/1199 [44:18<20:52,  3.85s/it][09:56:58:381661]:[Core] ManagedContainerBase.h(329)::checkExistsWithMessage : <Lighting Layout>::getObjectByHandle : Unknown Lighting Layout managed object handle :  . Aborting
#  81%|████████████████████████████████████████████████████████████████████████████████████████████████████████▊                        | 974/1199 [48:37<09:01,  2.41s/it][10:01:17:578803]:[Core] ManagedContainerBase.h(329)::checkExistsWithMessage : <Lighting Layout>::getObjectByHandle : Unknown Lighting Layout managed object handle :  . Aborting
#  90%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋             | 1074/1199 [52:57<05:03,  2.43s/it][10:05:38:276716]:[Core] ManagedContainerBase.h(329)::checkExistsWithMessage : <Lighting Layout>::getObjectByHandle : Unknown Lighting Layout managed object handle :  . Aborting
#  98%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎  | 1174/1199 [57:45<02:01,  4.85s/it][10:10:25:352180]:[Core] ManagedContainerBase.h(329)::checkExistsWithMessage : <Lighting Layout>::getObjectByHandle : Unknown Lighting Layout managed object handle :  . Aborting
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1199/1199 [58:44<00:00,  2.53s/it]==================================================
# Averaged metrics
# ==================================================
# episode_count: 1199.0
# does_want_terminate: 1.0
# num_steps: 100.30108423686406
# find_object_phase_success: 0.0
# pick_object_phase_success: 0.0
# find_recep_phase_success: 0.0
# overall_success: 0.0
# partial_success: 0.0
# ==================================================
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1199/1199 [58:44<00:00,  2.94s/it]
# Metrics:
#  {'episode_count': 1199.0, 'does_want_terminate': 1.0, 'num_steps': 100.30108423686406, 'find_object_phase_success': 0.0, 'pick_object_phase_success': 0.0, 'find_recep_phase_success': 0.0, 'overall_success': 0.0, 'partial_success': 0.0}


# minival

# GT
# ==================================================
# Averaged metrics
# ==================================================
# episode_count: 10.0
# does_want_terminate: 0.3
# num_steps: 939.2
# find_object_phase_success: 0.5
# pick_object_phase_success: 0.4
# find_recep_phase_success: 0.3
# overall_success: 0.1
# partial_success: 0.325
# ==================================================

# Base
# ==================================================
# Averaged metrics
# ==================================================
# episode_count: 10.0
# does_want_terminate: 0.2
# num_steps: 1090.3
# find_object_phase_success: 0.3
# pick_object_phase_success: 0.1
# find_recep_phase_success: 0.0
# overall_success: 0.0
# partial_success: 0.1
# ==================================================