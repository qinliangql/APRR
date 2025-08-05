
git submodule update --init --recursive src/third_party/detectron2 \
        src/home_robot/home_robot/perception/detection/detic/Detic \
        src/third_party/contact_graspnet 
pip install -e src/third_party/detectron2 
pip install -r src/home_robot/home_robot/perception/detection/detic/Detic/requirements.txt 
pip install -e src/home_robot 

# mkdir -p src/home_robot/home_robot/perception/detection/detic/Detic/models
# wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
#         -O src/home_robot/home_robot/perception/detection/detic/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
#         --no-check-certificate

# mkdir -p data/checkpoints 
# cd data/checkpoints 
# wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ovmm_baseline_home_robot_challenge_2023_v0.2.zip \
#         -O ovmm_baseline_home_robot_challenge_2023_v0.2.zip \
#         --no-check-certificate 
# unzip ovmm_baseline_home_robot_challenge_2023_v0.2.zip -d ovmm_baseline_home_robot_challenge_2023_v0.2 
# rm ovmm_baseline_home_robot_challenge_2023_v0.2.zip 