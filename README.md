# Active Perception Meets Rule-Guided RL: A Two-Phase Approach for Precise Object Navigation in Complex Environments

## Video Demos

### Video 1: Comparative Analysis with Baseline Methods
[https://github.com/user-attachments/assets/9693434a-edae-4b10-9224-acae015ade38](https://github.com/user-attachments/assets/f61c2fba-6c2e-41f3-b5d8-203bc15f98a2)

### Video 2: Active Navigation in Last-Mile Phase
[https://github.com/user-attachments/assets/f80bb6cc-46da-4c6e-8b4a-6f4dbbfa13b1](https://github.com/user-attachments/assets/1ce6dc30-5a94-4ed1-b0ac-bb2bd72d4ffa)


## Environment Setup

This repository presents a streamlined implementation of the original Active Perception with Rule-Guided Reinforcement Learning (APRR) approach. Redundant components from the original formulation have been omitted to enhance deployability and interpretability, while maintaining performance.

1.Follow the environment configuration guidelines for HomeRobot as specified in the official documentation:
[HomeRobot Setup Instructions](https://github.com/facebookresearch/home-robot/blob/home-robot-ovmm-challenge-2024/docs/challenge.md)

2.Alternatively, utilize the Docker image provided by the HomeRobot project:
[HomeRobot Docker Repository](https://hub.docker.com/r/fairembodied/habitat-challenge/tags)

3.Ensure the Conda environment named "home-robot" is properly initialized:
```bash
conda activate home-robot
cd home-robot/src/home-robot
pip install -e .
```

4.Install YOLO-World and SAM (Segment Anything Model) following the guidelines in the Ultralytics documentation:
[Ultralytics Installation Guide](https://docs.ultralytics.com/zh/models/yolo-world/)


5.Configure simulation
```bash
cd home-robot
bash set.sh
```

## Evaluation
The pre-trained model checkpoint is available for download at:
[Model Checkpoint (BaiduNetDisk)](https://pan.baidu.com/s/1bp8iO3GBbmfE9bZ93IKQng?pwd=uw53)


To evaluate the baseline method:
```bash
bash runb.sh
```

To evaluate the APRR framework:
```bash
bash run_rl.sh
```

## Training Procedures

For detailed training protocols, please refer to the official HomeRobot documentation: [HomeRobot Training Guidelines](https://github.com/facebookresearch/home-robot/blob/home-robot-ovmm-challenge-2024/docs/challenge.md)

