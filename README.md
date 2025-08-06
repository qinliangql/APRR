# Active Perception Meets Rule-Guided RL: A Two-Phase Approach for Precise Object Navigation in Complex Environments

## Video Demos

### Comparative Analysis with Baseline Methods
[https://github.com/user-attachments/assets/9693434a-edae-4b10-9224-acae015ade38](https://github.com/user-attachments/assets/f61c2fba-6c2e-41f3-b5d8-203bc15f98a2)

### Active Navigation in Last-Mile Phase
[https://github.com/user-attachments/assets/f80bb6cc-46da-4c6e-8b4a-6f4dbbfa13b1](https://github.com/user-attachments/assets/1ce6dc30-5a94-4ed1-b0ac-bb2bd72d4ffa)


## Setup

Follow the Home-robot Environment Setup [HomeRobot](https://github.com/facebookresearch/home-robot/blob/home-robot-ovmm-challenge-2024/docs/challenge.md)

You can directly user the docker image [HomeRobot Docker](https://hub.docker.com/r/fairembodied/habitat-challenge/tags)

Make sure the conda env "home-robot" is ready

```
conda activate home-robot
cd home-robot/src/home-robot
pip install -e .
```

Install yoloworld and sam ref [ultralytics](https://docs.ultralytics.com/zh/models/yolo-world/)


Set Sim config
```
cd home-robot
bash set.sh
```

## Eval

Run the baseline
```
bash runb.sh
```

Run APRR
```
bash run_rl.sh
```

## Train

Ref the Home-Robot Train [HomeRobot](https://github.com/facebookresearch/home-robot/blob/home-robot-ovmm-challenge-2024/docs/challenge.md)

