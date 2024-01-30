import argparse
from itertools import count
import random
import time

import gym
from gym.utils.save_video import save_video
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import numpy as np
import scipy.optimize

import torch
from torch import nn
from torch import optim
from torch.distributions.normal import Normal
from torch.autograd import Variable

import os
import logging
from distutils.util import strtobool

logging.basicConfig(filemode='w', format="%(asctime)s-%(name)s-%(levelname)s-%(message)s", level=logging.INFO)

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--env-name', default='Humanoid-v4', metavar='G',
                        help='name of the environment to run')
    parser.add_argument('--model-file-name', type=str, default='trpo-humanoid-gaussian')
    parser.add_argument('--model-version', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='the learning rate of optimizer')
    parser.add_argument('--seed', type=int, default=2024, metavar='N',
                        help='random seed (default: 2024)')
    parser.add_argument('--total-timesteps', type=int, default=2000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False')
    parser.add_argument('--cuda', type=lambda x : bool(strtobool(x)), default=True, nargs="?", const=True,
                        help='if toggled, cuda will be enabled by defaul')
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="trpo-humanoid-gaussian",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--train-flag", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="whether to train model")
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 10)')

    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument('--num-seeds', type=int, default=3,
                        help="the number of random seeds")
    parser.add_argument("--num-steps", type=int, default=2048,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.3,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization regression (default: 1e-3)')
    parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                        help='max kl value (default: 1e-2)')
    parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                        help='damping (default: 1e-1)')

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def make_env(env_name, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_name, render_mode='rgb_array_list')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs : np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __int__(self, envs):
        super(Agent, self).__int__()

        if args.train_flag:
            obs_shape = envs.single_observation_space.shape
            action_shape = envs.single_action_space.shape
        else:
            obs_shape = envs.observation_space.shape
            action_shape = envs.action_space.shape

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(action_shape)), std=0.01)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(), probs.entropy().sum(), self.critic(x)


def test(state_path, v_save_path):
    env = make_env(args.env_name, args.seed, 0, capture_video=False, run_name=run_name)()
    model = Agent(env)
    model.load_state_dict(torch.load(state_path, map_location=torch.device(device)))
    recorder = VideoRecorder(env, path=v_save_path)
    obs, infos = env.reset()
    obs = torch.Tensor(obs).to(device)
    total_reward = 0
    for step_index in range(3000):
        obs = obs.reshape(1, -1)
        action, logprob, _, value = model.get_action_and_value(obs)
        action = action.reshape(-1, )
        next_obs, reward, done, _, infos = env.step(action.cpu().numpy())
        recorder.capture_frame()
        total_reward += reward
        if done:
            next_obs, infos = env.reset()
        obs = torch.Tensor(next_obs).to(device)
    env.close()

    return total_reward


if __name__ == '__main__':
    args = parse_args()
    run_name = f"{args.gym_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    logging.info("run_name: {}".format(run_name))

    video_path = './videos/'
    if not os.path.exists(video_path):
        os.makedirs(video_path)

    model_param_path = "../models/"
    if not os.path.exists((model_param_path)):
        os.makedirs(model_param_path)
    model_param_file = os.path.join(model_param_path, args.model_file_name)
    args.train_flag = False

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda() else "cpu")

