import time
import glob
import argparse
import sys

import gym
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import multiprocessing as mp

from PIL import Image
from scipy.signal import lfilter

import os
os.environ['OMP_NUM_THREADS'] = '1'

import logging
logging.basicConfig(filemode="w", format="%(asctime)s-%(name)s-%(levelname)s-%(message)s", level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='Breakout-v4', type=str, help='gym environment')
    parser.add_argument('--processes', default=20, type=int, help='number of processes to train with')
    parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
    parser.add_argument('--test', default=False, type=bool, help='sets lr=0, chooses most likely actions')
    parser.add_argument('--rnn-steps', default=20, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
    parser.add_argument('--tau', default=1.0, type=float, help='generalized advantage estimation discount')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--hidden', default=256, type=int, help='hidden size of GRU')
    return parser.parse_args()


discount = lambda x, gamma: lfilter([1], [1, -gamma], x[::-1])[::-1]


class NNPolicy(nn.Module):
    def __init__(self, channels, memsize, num_actions):
        super(NNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru = nn.GRUCell(32 * 5 * 5, memsize)
        self.critic_linear, self.actor_linear = nn.Linear(memsize, 1), nn.Linear(memsize, num_actions)

    def forward(self,inputs, train=True, hard=False):
        inputs, hx = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        hx = self.gru(x.view(-1, 32 * 5 * 5), (hx))
        return self.critic_linear(hx), self.actor_linear(hx), hx

    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar')
        step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts)
            step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
            if step is 0:
                logging.info("\tno saved models")
            else:
                logging.info("\tloaded model: {}".format(paths[ix]))
        return step


class ShareAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(ShareAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None : continue
                self.state[p]['shared_steps'] += 1
                self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1
        super.step(closure)


def const_func(args, values, logps, actions, rewards):
    np_values = values.view(-1).data.numpy()

    delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
    logpys = logps.gather(1, torch.tensor(actions).view(-1, 1))
    gen_adv_est = discount(delta_t, args.gamma * args.tau)
    policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()

    rewards[-1] += args.gamma * np_values[-1]
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:-1, 0]).pow(2).sum()

    entropy_loss = (-logpys * torch.exp(logpys)).sum()
    return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss


def train(shared_model, shared_optimizer, rank, args, info):
    env = gym.make(args.env)
    # env.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    model = NNPolicy(channels=1, memsize=args.hidden, num_actions=args.num_actions)
    obs = env.reset()
    obs = np.array(obs)
    img = Image.fromarray(obs)
    state = torch.tensor(img.resize((80, 80)))

    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, done = 0, 0, 0, True

    while info['frames'][0] <= 8e7 or args.test:
        model.load_state_dict(shared_model.state_dict())

        hx = torch.zeros(1, 256)
        values, logps, actions, rewards = [], [], [], []

        for step in range(args.rNN_steps):
            episode_length += 1
            value, logit, hx = model((state.view(1, 1, 80, 80), hx))
            logp = F.log_softmax(logit, dim=1)

            action = torch.exp(logp).multinomial(num_samples=1).data[0]
            state, reward, done, _ = env.step(action.numpy()[0])
            if args.render:
                env.render()

            obs = np.array(state)
            img = Image.fromarray(obs)
            state = torch.tensor(img.resize((80, 80)))
            reward = np.clip(reward, -1, 1)
            done = done or episode_length >= 1e4

            info['frames'].add_(1)
            num_frames = int(info['frames'].item())
            if num_frames % 2e6 == 0:
                logging.info(args, '\n\t{:.0f}M frames: saved model\n'.format(num_frames / 1e6))
                torch.save(shared_model.state_dict(), args.save_dir + 'model.{:0.f}.tar'.format(num_frames / 1e6))

            if done:
                info['episodes'] += 1
                if info['episodes'][0] == 1:
                    interp = 1
                else:
                    interp = 1 - args.herizon
                info['run_epr'].mul_(1 - interp).add_(interp * epr)
                info['run_loss'].mul_(1 - interp).add_(interp * eploss)

            if rank == 0 and time.time() - last_disp_time > 60:
                elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                logging.info(args, 'time {}, episodes {:.0f}, frames {:.1f}M, mean epr {:.2f}, run loss {:.2f}'.format(
                    elapsed, info['episodes'].item(), num_frames / 1e6, info['run_epr'].item(), info['run_loss'].item()))
                last_disp_time = time.time()

            if done:
                episode_length, epr, eploss = 0, 0, 0
                obs = env.reset()
                img = Image.fromarray(obs)
                state = torch.tensor(img.resize((80, 80)))
                state = torch.tensor(state)

            values.append(value)
            logps.append(logp)
            actions.append(action)
            rewards.append(reward)
        if done:
            next_value = torch.zeros(1, 1)
        else:
            next_value = model((state.unsqueeze(0), hx))[0]
        values.append(next_value.detach())

        loss = const_func(args, torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards))
        eploss += loss.item()
        shared_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)

        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad =  param.grad
        shared_optimizer.step()




if __name__ == '__main__':
    if sys.version_info[0] > 2:
        mp.set_start_method('spawn')
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        raise "Must be using Python 3 with linux!"

    args = get_args()
    args.save_dir = '{}/'.format(args.env.lower())
    if args.render:
        args.processes = 1
        args.test = True
    if args.test:
        args.lr = 0
    args.num_actions = gym.make(args.env).action_space.shape
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    torch.manual_seed(args.seed)
    shared_model = NNPolicy(channels=1, memsize=args.hidden, num_actions=args.num_actions).share_memory()
    shared_optimizer = ShareAdam(shared_model.parameters(), lr=args.lr)

    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}
    info['frames'] += shared_model.try_load(args.save_dir) * 1e6
    if int(info['frames'].item()) == 0:
        logging.info(args)

    processes = []
    for rank in range(args.processes):
        p = mp.Process(target=train, args=(shared_model, shared_optimizer, rank, args, info))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()