import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as Optim
import numpy as np
from torch.distributions import MultivariateNormal

class Actor(nn.Module):
  def __init__(self,input_dims,fc1_dims,fc2_dims,action_dims,device):
    super(Actor,self).__init__()

    self.fc1 = nn.Linear(input_dims,fc1_dims)
    self.ln1 = nn.LayerNorm(fc1_dims)

    self.fc2 = nn.Linear(fc1_dims,fc2_dims)
    self.ln2 = nn.LayerNorm(fc2_dims)

    self.mu = nn.Linear(fc2_dims,action_dims)

    self.device = device
    self.to(self.device)

  def forward(self,state):
    state = torch.Tensor(state).to(self.device)

    x = F.relu(self.ln1(self.fc1(state)))
    x = F.relu(self.ln2(self.fc2(x)))

    action_mu = torch.tanh(self.mu(x))

    return action_mu


class Critic(nn.Module):
  def __init__(self,input_dims,fc1_dims,fc2_dims,action_dims,device):
    super(Critic,self).__init__()

    self.fc1 = nn.Linear(input_dims,fc1_dims)
    self.ln1 = nn.LayerNorm(fc1_dims)

    self.fc2 = nn.Linear(fc1_dims,fc2_dims)
    self.ln2 = nn.LayerNorm(fc2_dims)

    self.action_value_layer = nn.Linear(action_dims,fc2_dims)

    self.q = nn.Linear(fc2_dims,1)

    self.device = device
    self.to(self.device)

  def forward(self,state,action):
    state = torch.Tensor(state).to(self.device)

    try:
      action = torch.Tensor(action).to(self.device)
    except:
      action = action

    state_value = F.relu(self.ln1(self.fc1(state)))
    state_value = self.ln2(self.fc2(state_value))

    action_value = F.relu(self.action_value_layer(action))

    state_action_value = F.relu(torch.add(state_value,action_value))
    state_action_value = self.q(state_action_value)

    return state_action_value


class Agent:
  def __init__(self,input_dims,fc1_dims,fc2_dims,action_dims,actor_alpha,critic_alpha,gamma,start_action_std,min_action_std,action_std_decay_rate,
               action_update_delay = 2,n_update_iter = 300,target_action_noise_clip = 0.25,tau=0.001,batch_size=128,buffer_size = 100000,device='cpu'):


    self.batch_size = batch_size
    self.gamma = gamma
    self.buffer_size = buffer_size
    self.action_dims = action_dims
    self.input_dims = input_dims
    self.device = torch.device(device)
    self.tau = tau
    self.n_update_iter = n_update_iter
    self.action_update_delay = action_update_delay

    self.actor = Actor(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,action_dims=action_dims,device=device)
    self.target_actor = Actor(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,action_dims=action_dims,device=device)

    self.critic_1 = Critic(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,action_dims=action_dims,device=device)
    self.target_critic_1 = Critic(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,action_dims=action_dims,device=device)

    self.critic_2 = Critic(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,action_dims=action_dims,device=device)
    self.target_critic_2 = Critic(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,action_dims=action_dims,device=device)

    self.critic_optimizer_1 = Optim.Adam(self.critic_1.parameters(),lr=critic_alpha)
    self.critic_optimizer_2 = Optim.Adam(self.critic_2.parameters(),lr=critic_alpha)
    self.actor_optimizer = Optim.Adam(self.actor.parameters(),lr=actor_alpha)
    self.mse_loss = nn.MSELoss()

    self.action_std = start_action_std
    self.action_var = torch.full((self.action_dims,), self.action_std * self.action_std).to(device)
    self.action_std_decay_rate = action_std_decay_rate
    self.min_action_std = min_action_std
    self.target_action_noise_clip = target_action_noise_clip

    self.learn_counter = 0
    self.buffer_counter = 0

    self.state_buffer = np.zeros((self.buffer_size, input_dims))
    self.next_state_buffer = np.zeros((self.buffer_size, input_dims))
    self.action_buffer = np.zeros((self.buffer_size,action_dims))
    self.reward_buffer = np.zeros((self.buffer_size,))
    self.terminal_buffer = np.zeros((self.buffer_size,), dtype=np.float32)

    self.update_network_parameters(tau=1)


  def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params_1 = self.critic_1.named_parameters()
        critic_params_2 = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params_1 = self.target_critic_1.named_parameters()
        target_critic_params_2 = self.target_critic_2.named_parameters()

        critic_state_dict_1 = dict(critic_params_1)
        critic_state_dict_2 = dict(critic_params_2)
        actor_state_dict = dict(actor_params)
        target_critic_dict_1 = dict(target_critic_params_1)
        target_critic_dict_2 = dict(target_critic_params_2)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict_1:
            critic_state_dict_1[name] = tau*critic_state_dict_1[name].clone() +(1-tau)*target_critic_dict_1[name].clone()
        self.target_critic_1.load_state_dict(critic_state_dict_1)

        for name in critic_state_dict_2:
            critic_state_dict_2[name] = tau*critic_state_dict_2[name].clone() +(1-tau)*target_critic_dict_2[name].clone()
        self.target_critic_2.load_state_dict(critic_state_dict_2)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)




  def set_action_std(self, new_action_std):
    self.action_std = new_action_std
    self.action_var = torch.full((self.action_dims,), self.action_std * self.action_std).to(self.device)



  def decay_std(self):
    print("##################################################")
    print("Changing std")

    new_action_std = self.action_std - self.action_std_decay_rate
    new_action_std = round(new_action_std,4)

    if new_action_std <= self.min_action_std:
      self.action_std = self.min_action_std
    else:
      self.action_std = new_action_std

    print("New Action std:",self.action_std)
    self.set_action_std(self.action_std)
    print("##################################################")


  def store_transitions(self,state,action,reward,new_state,done):

    index = self.buffer_counter % self.buffer_size

    self.state_buffer[index] = state
    self.action_buffer[index] = action
    self.reward_buffer[index] = reward
    self.next_state_buffer[index] = new_state
    self.terminal_buffer[index] = float(1- done)

    self.buffer_counter += 1


  def choose_action(self,state):
    #state = torch.Tensor(state).to(self.device)

    action_mean = self.actor.forward(state)
    action_mean = action_mean.float()

    cov_mat = torch.diag(self.action_var).unsqueeze(dim=0).to(self.device)
    cov_mat = cov_mat.float()

    dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
    action = dist.sample()
    action = torch.clamp(action,-1,1)

    return action.detach().cpu().numpy().flatten()


  def update(self):

    if self.buffer_counter >= self.batch_size:
      for n in range(self.n_update_iter):
        max_mem = min(self.buffer_counter, self.buffer_size)

        batch_indices = np.random.choice(max_mem, self.batch_size)

        state_batch = self.state_buffer[batch_indices]
        action_batch = self.action_buffer[batch_indices]
        reward_batch = self.reward_buffer[batch_indices]
        next_state_batch = self.next_state_buffer[batch_indices]
        terminal_batch = self.terminal_buffer[batch_indices]

        reward_batch = torch.Tensor(reward_batch).to(self.device).reshape((self.batch_size,1))
        terminal_batch = torch.Tensor(terminal_batch).to(self.device).reshape((self.batch_size,1))

        next_action_batch = self.target_actor.forward(next_state_batch).detach()
        target_action_noise = torch.randn(next_action_batch.shape).to(self.device)
        target_action_noise = torch.clamp(target_action_noise,-self.target_action_noise_clip,self.target_action_noise_clip)
        next_action_batch = next_action_batch + target_action_noise
        next_action_batch = torch.clamp(next_action_batch,-1,1)

        next_critic_value_1 = self.target_critic_1.forward(next_state_batch,next_action_batch).detach()
        next_critic_value_2 = self.target_critic_2.forward(next_state_batch,next_action_batch).detach()
        next_critic_value = torch.min(next_critic_value_1,next_critic_value_2)

        critic_value_1 = self.critic_1.forward(state_batch,action_batch)
        critic_value_2 = self.critic_2.forward(state_batch,action_batch)

        critic_targets = reward_batch + self.gamma*next_critic_value*terminal_batch

        self.critic_optimizer_1.zero_grad()
        critic_loss_1 = self.mse_loss(critic_value_1,critic_targets)
        critic_loss_1.backward()
        self.critic_optimizer_1.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss_2 = self.mse_loss(critic_value_2,critic_targets)
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        if n % self.action_update_delay == 0:
          mu = self.actor.forward(state_batch)
          actor_loss = -self.critic_1.forward(state_batch,mu)
          self.actor_optimizer.zero_grad()
          actor_loss.mean().backward()
          self.actor_optimizer.step()

          self.update_network_parameters()
