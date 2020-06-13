# Deep Reinforcement Learning with CartPole in Pytorch
## About
This repository explores 3 different Reinforcement Learning Algorithms using Deep Learning in Pytorch. The methods used here include Deep Q Learning (DQN), Policy Gradient Learning (REINFORCE), and Advantage Actor-Critic (A2C). 

<p align="center">
  <img src="saved_dqn_cartpole/movie.gif" width="280">
  <img src="saved_reinforce_cartpole/movie.gif" width="280">
  <img src="saved_a2c_cartpole/movie.gif" width="280">
</p>

## Models
### Model: DQN
The Deep Q-Network (DQN) is implemented as a simple feedforward network with two hidden layers of size 128 and 64, respectively. 
<p align="center">
  <img src="saved_dqn_cartpole/movie.gif" width="400">
  <img src="saved_dqn_cartpole/plot_rewards.png" width="450">
</p>

### Model: REINFORCE
The Policy-Gradient Network (REINFORCE) is implemented as a simple feedforward network with a single hidden layer of size 128 and output size equal to that of the action space. A dropout layer of 0.6 is placed in the intermediate layer.
<p align="center">
  <img src="saved_reinforce_cartpole/movie.gif" width="400">
  <img src="saved_reinforce_cartpole/plot_rewards.png" width="450">
</p>

### Model: A2C
The Advantage Actor-Critic (A2C) consists of 2 modules, an actor and a critic. The actor has a hidden layer of size 128 and output size equal to that of the action space. A dropout layer of 0.7 is placed in the intermediate layer. The critic has a hidden layer of size 128 and output size of 1. 
<p align="center">
  <img src="saved_a2c_cartpole/movie.gif" width="400">
  <img src="saved_a2c_cartpole/plot_rewards.png" width="450">
</p>


## Layout
Files are named in the format name-of-model.py and corresponding folders are name-of-model/, containing plots, models, and videos of trained agents performing the CartPole task.

## Run

## References
Refer below for some fantastic tutorials on the topic, without which this code would not be possible: 
* https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
* https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf
* https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
