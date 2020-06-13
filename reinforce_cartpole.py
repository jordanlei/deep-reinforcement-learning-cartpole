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

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0) #cuda device
parser.add_argument('--verbose', type=int, default=1) #printing preferences
parser.add_argument('--load', type=bool, default = False) #if loading an existing model
parser.add_argument('--save', type=bool, default = False) #if saving an existing model
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
    x = self.l1(x)
    x = F.relu(self.dropout(x))
    x = self.l2(x)
    return self.softmax(x)

class Runner():
  def __init__(self, net, optimizer, gamma = 0.99, logs = "reinforce_cartpole"):
    self.net = net
    self.optimizer = optimizer
    self.gamma = gamma
    self.writer = SummaryWriter(logs)
    self.logs = logs
  
  def env_step(self, action):
    state, reward, done, log = env.step(action)
    return torch.FloatTensor([state]).to(device), torch.FloatTensor([reward]).to(device), done, log

  def select_action(self, state):
    #convert state to tensor
    probs = self.net(torch.FloatTensor([state]).to(device))
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
    
    rewards = torch.FloatTensor(rewards)
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
      running_reward = running_reward * self.gamma + time * (1-self.gamma)
      loss = self.update_policy()
      self.writer.add_scalar("Loss", loss, episode)
      self.writer.add_scalar("Reward", rewards, episode)
      if len(smoothed_reward) > smooth: 
        smoothed_reward = smoothed_reward[-1*smooth: -1]
      self.writer.add_scalar("Mean Reward", np.mean(smoothed_reward), episode)
      if episode % 20 == 0: 
        print("\tEpisode {} \t Final Reward {:.2f} \t Average Reward: {:.2f}".format(episode, rewards, running_reward))
  
  def run(self):
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
      plt.title("Policy Gradient Agent")
      ims.append([im])

    print("\tTotal Reward: ", rewards)
    env.close()
    print("\tSaving Animation ...")
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    ani.save('%s-movie.avi'%self.logs)
    # animation.save('animation.gif', writer='PillowWriter', fps=2)

  def save(self): 
    torch.save(self.net.state_dict(),'%s-model.pt'%self.logs)

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

    if args.save: 
        print("[Save]\tSaving Model ...")
        runner.save()

    if "run" in args.runtype:
        print("[Run]\tRunning Simulation ...")
        runner.run()
    

    print("[End]\tDone. Congratulations!")

if __name__ == '__main__':
    main()