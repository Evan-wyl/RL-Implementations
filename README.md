# run-rl
There are some implementation details of Reinforcement Learning algorithms, including PPO||A3C||A2C||SAC||GAIL.



| 算法 | 代码地址                                                    | 说明文档                                                     |
| ---- | ----------------------------------------------------------- | ------------------------------------------------------------ |
| PPO  | [code](https://github.com/Evan-wyl/run-rl/tree/master/ppo)  | [details](https://www.yuque.com/u2274123/xrrca1/huvxggkboeso7sr5) |
| TRPO | [code](https://github.com/Evan-wyl/run-rl/tree/master/TRPO) | [details](https://www.yuque.com/u2274123/xrrca1/uo39n650cyyil6gg) |
| A3C  |                                                             |                                                              |
| A2C  |                                                             |                                                              |
| SAC  |                                                             |                                                              |
| GAIL |                                                             |                                                              |
| PPG  |                                                             |                                                              |

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
pip install mujoco
pip install moviepy
pip install glx
conda install ffmpeg -c conda-forge

pip install autorom(install command)
AutoROM --accept-license(execute command)

# isaacgym
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -zxvf IsaacGym_Preview_4_Package.tar.gz
cd issacgym
pip install -e .

https://developer.nvidia.com/isaac-gym/download
```







## 参考文献

[The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
