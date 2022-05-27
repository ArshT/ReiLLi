import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.multiprocessing as mp
from collections import namedtuple
import os
import warnings
warnings.filterwarnings("ignore")
import gym

from algorithms.PPO.ppo import Agent_Discrete
from algorithms.PPO.ppo_continuous import Agent_Continuous


Msg = namedtuple('Msg', ['agent', 'reached', 'avg_reward'])



class Memory():
  def __init__(self,input_dims,device,update_timestep,num_agents,continuous,action_dims=None):
    self.device = torch.device(device)

    self.states = torch.zeros((num_agents*update_timestep,input_dims)).to(self.device).share_memory_()
    self.logprobs = torch.zeros((num_agents*update_timestep)).to(self.device).share_memory_()

    if not continuous:
      self.actions = torch.zeros((num_agents*update_timestep)).to(self.device).share_memory_()
    else:
      self.actions = torch.zeros((num_agents*update_timestep,action_dims)).to(self.device).share_memory_()

    self.disReturn = torch.zeros(update_timestep*num_agents).to(self.device).share_memory_()

class Agent(mp.Process):
  def __init__(self,name,env_name,pipe_end,max_timesteps_per_ep,update_timestep,agent,memory,device,seed):
    mp.Process.__init__(self, name=name)

    self.memory = memory
    self.agent = agent
    self.name = name
    self.pipe_end = pipe_end
    self.max_timesteps_per_ep = max_timesteps_per_ep
    self.update_timestep = update_timestep
    self.device = torch.device(device)

    self.env = gym.make(env_name)
    self.env.seed(seed)
    self.input_dims = self.env.observation_space.shape[0]

  def choose_action(self,state):
    action = self.agent.choose_action(state)
    return action

  def store_transitions(self,state,reward,terminal):
    self.agent.store_transitions_multi(state,reward,terminal)

  def experience_to_tensor(self):
    state_tensor,action_tensor,logprob_tensor,G_tensor = self.agent.experience_to_tensor(self.update_timestep)
    return state_tensor,action_tensor,logprob_tensor,G_tensor


  def add_experience_to_memory(self, state_tensor, action_tensor,logprob_tensor, disReturn_tensor):
    start_idx = int(self.name) * self.update_timestep
    end_idx = start_idx + self.update_timestep


    self.memory.states[start_idx:end_idx] = state_tensor
    self.memory.actions[start_idx:end_idx] = action_tensor
    self.memory.logprobs[start_idx:end_idx] = logprob_tensor
    self.memory.disReturn[start_idx:end_idx] = disReturn_tensor


  def run(self):
    print("Agent {} started, Process ID {}".format(self.name, os.getpid()))

    total_timesteps = 0
    scores = []

    for episode_number in range(1000):
      done = False
      score = 0
      observation = self.env.reset()
      timestep = 0

      while not done and timestep <= self.max_timesteps_per_ep:
        action = self.choose_action(observation)
        observation_,reward,done,_ = self.env.step(action)
        score += reward
        self.store_transitions(observation,reward,done)
        observation = observation_
        timestep += 1
        total_timesteps += 1

      scores.append(score)
      if total_timesteps >= self.update_timestep:
        break

    avg_score = sum(scores) / len(scores)

    state_tensor,action_tensor,logprob_tensor,G_tensor = self.experience_to_tensor()
    self.add_experience_to_memory(state_tensor,action_tensor,logprob_tensor,G_tensor)

    self.agent.reward_memory = []
    self.agent.terminal_memory = []
    self.agent.state_memory = torch.empty((1,self.input_dims))
    self.agent.action_memory = torch.empty((1,)).to(self.device)
    self.agent.logprob_memory = torch.empty((1,)).to(self.device)

    msg = Msg(int(self.name),True,avg_score)
    self.pipe_end.send(msg)

class PPO_multi_agent:
  def __init__(self,env_name,num_agents,max_rounds,max_timesteps_per_ep,update_timestep,solved_reward,fc1_dims,fc2_dims,actor_alpha,critic_alpha,
               gamma,K_epochs,eps_clip,num_test_episodes,continuous,render,start_action_std=None,action_std_decay_rate=None,min_action_std=None,action_std_decay_round=None):

    self.device = torch.device('cpu')
    self.env_name = env_name
    self.num_agents = num_agents
    self.max_rounds = max_rounds
    self.max_timesteps_per_ep = max_timesteps_per_ep
    self.update_timestep = update_timestep
    self.solved_reward = solved_reward
    self.num_test_episodes = num_test_episodes
    self.render = render

    self.action_std_decay_round = action_std_decay_round

    sample_env = gym.make(env_name)
    input_dims = sample_env.observation_space.shape[0]
    try:
      action_dims = sample_env.action_space.n
    except:
      action_dims = sample_env.action_space.shape[0]
    del sample_env

    self.memory = Memory(num_agents = num_agents, update_timestep = update_timestep,input_dims = input_dims, device=self.device,continuous=continuous,action_dims=action_dims)

    if not continuous:
      self.ppo = Agent_Discrete(input_dims = input_dims,n_actions = action_dims,actor_alpha = actor_alpha,critic_alpha = critic_alpha,
                                eps_clip = eps_clip, K_epochs = K_epochs,fc1_dims=fc1_dims,fc2_dims=fc2_dims,gamma=gamma,device=self.device)

    else:
      self.ppo = Agent_Continuous(input_dims=input_dims,action_dims=action_dims,actor_alpha=actor_alpha,critic_alpha=critic_alpha,gamma=gamma,eps_clip=eps_clip,
                                  K_epochs=K_epochs,start_action_std=start_action_std,action_std_decay_rate=action_std_decay_rate,min_action_std=min_action_std,
                                  fc1_dims=fc1_dims,fc2_dims=fc2_dims,device=self.device)


  def train(self):
    print("#################################")
    print(self.env_name)
    print("Number of Agents: {}".format(self.num_agents))
    print("#################################\n")

    for round in range(self.max_rounds):
      print("Round:",round+1)

      agents = []
      pipes = []

      agents_completed = [False] * self.num_agents

      solved_flag = False

      if self.action_std_decay_round:
        if (round+1)%self.action_std_decay_round == 0:
          self.ppo.decay_std()

      for agent_id in range(self.num_agents):
        p_start,p_end = mp.Pipe()

        agent = Agent(name = str(agent_id), memory = self.memory,pipe_end = p_end,env_name = self.env_name,max_timesteps_per_ep = self.max_timesteps_per_ep,
                      update_timestep=self.update_timestep,agent=self.ppo,seed=agent_id*10,device=self.device)

        agent.start()
        agents.append(agent)
        pipes.append(p_start)

      avg_score = 0
      while True:
        for i,conn in enumerate(pipes):
          if conn.poll():
            msg = conn.recv()
            if type(msg).__name__ == 'Msg':
              avg_score += msg.avg_reward
              agents_completed[i] = True

        if False not in agents_completed:
          self.ppo.learn_multi(self.memory,round + 1)
          agents_completed = [False] * self.num_agents

          for agent in agents:
            agent.terminate()

          print("#####################################################################################")
          print("Average Score:", avg_score / self.num_agents)

          if (avg_score / self.num_agents) >= self.solved_reward:
            print("########SOLVED!!!##########")
            solved_flag = True
            break

          print("Training Batch Completed")
          print("#####################################################################################")
          print()
          break

      if solved_flag == True:
        break

  def test(self):
    total_score = 0
    test_env = gym.make(self.env_name)

    for i in range(self.num_test_episodes):
      done = False
      observation = test_env.reset()
      score = 0
      while not done:
        action = self.ppo.choose_action(observation)
        observation_, reward, done, info = test_env.step(action)
        observation = observation_
        score += reward
        if self.render:
          test_env.render()
      total_score += score
    print("Average Score:", total_score/self.num_test_episodes)
