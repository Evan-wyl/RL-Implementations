# RL Implementations

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![MIT License](https://img.shields.io/badge/license-Apache-green.svg)](https://opensource.org/licenses/MIT) [![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)



### Requirements

```python
# 创建虚拟环境
conda create -n py3.8 python=3.8

# 数据处理相关工具包
pip install numpy
pip install pandas
pip install matplotlib

#ml相关工具包
pip install torch
pip install torchsummary
pip install torchinfo
pip install scikit-learn
conda install cudnn

# 训练结果可视化工具包
pip install tensorboard
pip install wandb

# opencv相关工具包
pip install opencv-python-headless
pip install opencv-contrib-python-headless

# gym相关包
pip install gym
pip install stable_baselines3
pip install ale-py
pip install moviepy
pip install glx
conda install ffmpeg -c conda-forge

pip install autorom(install command)
AutoROM --accept-license(execute command)

# Mujoco
pip install mujoco

# isaacgym
conda create -n rlgpu python=3.7

## Nvidia Driver version
apt install nvidia-driver-525-server
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

wget https://developer.nvidia.com/isaac-gym-preview-4
tar -zxvf IsaacGym_Preview_4_Package.tar.gz
cd issacgym
pip install -e .
```



### Related Project

| Project                                                      | Including Algorithms              |
| ------------------------------------------------------------ | --------------------------------- |
| [A3C-GRU](https://github.com/pranz24/A3C-GRU) [baby-a3c](https://github.com/greydanus/baby-a3c) | A3C                               |
| [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) | A2C<br />PPO<br />ACKTR<br />GAIL |
| [pytorch-trpo](https://github.com/ikostrikov/pytorch-trpo) [trpo](https://github.com/pat-coady/trpo) | TRPO                              |
| [pytorch_sac](https://github.com/denisyarats/pytorch_sac) [sac](https://github.com/haarnoja/sac) | SAC                               |
| [BCQ](https://github.com/sfujim/BCQ)                         | BCQ                               |
| [CQL](https://github.com/aviralkumar2907/CQL)                | CQL                               |
| [batch_rl](https://github.com/google-research/batch_rl)      |                                   |
| [seed_rl](https://github.com/google-research/seed_rl)        |                                   |



## Reference

[The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
