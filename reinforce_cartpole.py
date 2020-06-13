'''
Jordan Lei, 2020. Some code is based on the following sources:
   https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf
'''

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from collections import namedtuple
from itertools import count
from PIL import Image
from pyvirtualdisplay import Display
from IPython import display

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.distributions import Categorical
import ffmpeg
import seaborn as sns

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0) #cuda device
parser.add_argument('--verbose', type=int, default=1) #printing preferences
parser.add_argument('--load', type=bool, default = False) #if loading an existing model
parser.add_argument('--save', type=bool, default = False) #if saving an existing model
parser.add_argument('--plot', type=bool, default = True) #if plotting an existing model
parser.add_argument('--model', type=str, default='reinforce_cartpole/model.pt') #model - currently supports resnet and alexnet, with more to come
parser.add_argument('--runtype', type=str, default='train_run',
                        choices=('train', 'run', 'train_run')) #runtype: train only or train and validate
parser.add_argument('--lr', type=float, default=0.01)  #learning rate
parser.add_argument('--episodes', type=int, default=500) #number of episodes    
parser.add_argument('--gamma', type=float, default=0.99) #discount factor                                  
args = parser.parse_args()

#virtual display used to satisfy non-screen evironments (e.g. server)
# virtualdisplay = Display(visible=0, size=(1400, 900))
# virtualdisplay.start()

#setup environment
env = gym.make('CartPole-v0').unwrapped

#set the cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"]= "{}".format(args.device)
device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#policy gradient network
class Policy(nn.Module): 
  def __init__(self, in_size, out_size):
    super(Policy, self).__init__()
    self.in_size = in_size
    self.out_size = out_size

    self.l1 = nn.Linear(self.in_size, 128)
    self.l2 = nn.Linear(128, self.out_size)
    self.dropout = nn.Dropout(0.6)
    self.softmax = nn.Softmax(dim= 1)

    self.policy_history = Variable(torch.Tensor()).to(device)
    self.reward_episode = []

    self.reward_history = []
    self.loss_history = []

  def forward(self, x):
    # convert state to tensor

    x = Variable(torch.from_numpy(x).float().unsqueeze(0)).to(device) 
    x = self.l1(x)
    x = F.relu(self.dropout(x))
    x = self.l2(x)
    #softmax outputs a probability distribution over action space
    return self.softmax(x)

class Runner():
  def __init__(self, net, optimizer, gamma = 0.99, logs = "reinforce_cartpole"):
    self.net = net
    self.optimizer = optimizer
    self.gamma = gamma
    self.writer = SummaryWriter(logs)
    self.logs = logs
    self.plots = {"Loss": [], "Reward": [], "Mean Reward": []}
  

  def select_action(self, state):
    #convert state to tensor
    probs = self.net(state)
    c = Categorical(probs)
    action = c.sample()

    #place log probabilities into the policy history log\pi(a | s)
    if self.net.policy_history.dim()!= 0: 
      self.net.policy_history = torch.cat([self.net.policy_history, c.log_prob(action)])
    else: 
      self.net.policy_history = (c.log_prob(action))
    
    return action
  
  def update_policy(self): 
    R = 0 
    rewards = []

    #discount using gamma
    for r in self.net.reward_episode[::-1]: 
      R = r + self.gamma * R
      rewards.insert(0, R)
    
    #at this point, rewards is a proxy for Q(s, a)
    rewards = torch.FloatTensor(rewards)
    #normalize to reduce variance
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    #loss = - sum_t log(\pi(a|s)) * v_t
    loss = torch.sum(torch.mul(self.net.policy_history, Variable(rewards).to(device)).mul(-1), -1)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    #update reward history and loss histories
    self.net.loss_history.append(loss.item())
    self.net.reward_history.append(np.sum(self.net.reward_episode))
    #flush policy history
    self.net.policy_history = Variable(torch.Tensor()).to(device)
    self.net.reward_episode = []

    return loss

  
  def train(self, episodes = 200, smooth = 10): 
    running_reward = 10
    smoothed_reward = []
    for episode in range(episodes):
      state = env.reset()
      done = False
      rewards = 0

      for time in range(500): 
        action = self.select_action(state)
        state, reward, done, _ = env.step(action.data[0].item())
        rewards+= reward

        self.net.reward_episode.append(reward)
        if done: 
          break

      
      smoothed_reward.append(rewards)
      if len(smoothed_reward) > smooth: 
        smoothed_reward = smoothed_reward[-1*smooth: -1]

      running_reward = running_reward * self.gamma + time * (1-self.gamma)
      loss = self.update_policy()
      self.writer.add_scalar("Loss", loss, episode)
      self.writer.add_scalar("Reward", rewards, episode)
      self.writer.add_scalar("Mean Reward", np.mean(smoothed_reward), episode)
      
      self.plots["Loss"].append(loss)
      self.plots["Reward"].append(rewards)
      self.plots["Mean Reward"].append(np.mean(smoothed_reward))

      if episode % 20 == 0: 
        print("\tEpisode {} \t Final Reward {:.2f} \t Average Reward: {:.2f}".format(episode, rewards, np.mean(smoothed_reward)))
  
  def run(self):
    sns.set_style("dark")
    sns.set_context("poster")

    fig = plt.figure() 
    ims = []
    rewards = 0
    state = env.reset()
    for time in range(500):
      action = self.select_action(state) 
      state, reward, done, _ = env.step(action.data[0].item())
      rewards += reward

      if done:
        break
    
      im = plt.imshow(env.render(mode='rgb_array'), animated=True)
      plt.axis('off')
      plt.title("Policy Gradient Agent")
      ims.append([im])

    print("\tTotal Reward: ", rewards)
    env.close()
    print("\tSaving Animation ...")
    ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True,
                                    repeat_delay=1000)
    ani.save('%s-movie.avi'%self.logs, dpi = 300)
    # animation.save('animation.gif', writer='PillowWriter', fps=2)

  def plot(self):
    sns.set()
    sns.set_context("poster")

    plt.figure(figsize=(20, 16))
    plt.plot(np.arange(len(self.plots["Loss"])), self.plots["Loss"])
    plt.title("Policy Gradient Loss")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.savefig("%s/plot_%s.png"%(self.logs, "loss"))

    plt.figure(figsize=(20, 16))
    plt.plot(np.arange(len(self.plots["Reward"])), self.plots["Reward"], label="Reward")
    plt.plot(np.arange(len(self.plots["Mean Reward"])), self.plots["Mean Reward"], label = "Mean Reward")
    plt.legend()
    plt.title("Policy Gradient Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.savefig("%s/plot_%s.png"%(self.logs, "rewards"))

    # for key in self.plots.keys():
    #     data = self.plots[key]
    #     plt.figure(figsize=(20, 16))
    #     plt.plot(np.arange(len(data)), data)
    #     plt.title("Policy Gradient %s"%key)
    #     plt.xlabel("Episodes")
    #     plt.ylabel(key)
    #     plt.savefig("%s/plot_%s.png"%(self.logs, key))

  def save(self): 
    torch.save(self.net.state_dict(),'%s/model.pt'%self.logs)

def main(): 
    device_name = "cuda: %s"%(args.device) if torch.cuda.is_available() else "cpu"
    print("[Device]\tDevice selected: ", device_name)

    policy = Policy(env.observation_space.shape[0], env.action_space.n).to(device)
    
    #if we're loading a model
    if args.load: 
        policy.load_state_dict(torch.load(args.model))

    optimizer = optim.Adam(policy.parameters(), lr = args.lr)
    runner = Runner(policy, optimizer, gamma = args.gamma, logs = "reinforce_cartpole/%s" %time.time())
    
    if "train" in args.runtype:
        print("[Train]\tTraining Beginning ...")
        runner.train(args.episodes)

        if args.plot:
            print("[Plot]\tPlotting Training Curves ...")
            runner.plot()

    if args.save: 
        print("[Save]\tSaving Model ...")
        runner.save()
    
    
    if "run" in args.runtype:
        print("[Run]\tRunning Simulation ...")
        runner.run()
    

    print("[End]\tDone. Congratulations!")

if __name__ == '__main__':
    main()