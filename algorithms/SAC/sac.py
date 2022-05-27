import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Critic(nn.Module):
    def __init__(self,input_dims,fc1_dims,fc2_dims,action_dims,device='cpu'):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(input_dims,fc1_dims)
        self.ln1 = nn.LayerNorm(fc1_dims)

        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)

        self.action_value_layer = nn.Linear(action_dims,fc2_dims)

        self.q = nn.Linear(fc2_dims,1)

        self.device = device
        self.to(self.device)

        self.apply(weights_init_)

    def forward(self, state, action):

      state_value = F.relu(self.ln1(self.fc1(state)))
      state_value = self.ln2(self.fc2(state_value))

      action_value = F.relu(self.action_value_layer(action))

      state_action_value = F.relu(torch.add(state_value,action_value))
      state_action_value = self.q(state_action_value)

      return state_action_value


class GaussianPolicy(nn.Module):
    def __init__(self, input_dims,fc1_dims,fc2_dims,action_dims,device='cpu',log_std_min=-20,log_std_max=2,epsilon=1e-6):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(input_dims,fc1_dims)
        self.ln1 = nn.LayerNorm(fc1_dims)

        self.linear2 = nn.Linear(fc1_dims,fc2_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)

        self.mean_linear = nn.Linear(fc2_dims,action_dims)
        self.log_std_linear = nn.Linear(fc2_dims,action_dims)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.epsilon = epsilon

        self.device = torch.device(device)
        self.to(self.device)

        self.apply(weights_init_)


    def forward(self, state):
        x = F.relu(self.ln1(self.linear1(state)))
        x = F.relu(self.ln2(self.linear2(x)))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)

        return mean, log_std


    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log((1 - action.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean



class Agent(object):
    def __init__(self,input_dims,fc1_dims,fc2_dims,action_dims,gamma,tau,SAC_alpha,actor_alpha,critic_alpha,SAC_alpha_lr,batch_size,n_update_iter,tune_alpha=True,buffer_size = 1000000,device='cpu'):

        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.action_dims = action_dims
        self.tune_alpha = tune_alpha
        self.n_update_iter = n_update_iter

        self.device = torch.device(device)

        self.critic_1 = Critic(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,action_dims=action_dims,device=device)
        self.target_critic_1 = Critic(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,action_dims=action_dims,device=device)

        self.critic_2 = Critic(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,action_dims=action_dims,device=device)
        self.target_critic_2 = Critic(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,action_dims=action_dims,device=device)

        self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), lr=critic_alpha)
        self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), lr=critic_alpha)

        self.actor = GaussianPolicy(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,action_dims=action_dims,device=device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_alpha)

        self.learn_counter = 0
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_size, input_dims))
        self.next_state_buffer = np.zeros((self.buffer_size, input_dims))
        self.action_buffer = np.zeros((self.buffer_size,self.action_dims))
        self.reward_buffer = np.zeros((self.buffer_size,))
        self.terminal_buffer = np.zeros((self.buffer_size,), dtype=np.float32)

        if self.tune_alpha:
            self.target_entropy = -torch.prod(torch.Tensor(action_dims).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=SAC_alpha_lr)
            self.alpha = self.log_alpha.exp().detach()
        else:
            self.alpha = SAC_alpha

        self.update_network_parameters(tau=1)

    def store_transitions(self,state,action,reward,next_state,terminal):
      idx = self.buffer_counter % self.buffer_size

      self.state_buffer[idx] = state
      self.action_buffer[idx] = action
      self.reward_buffer[idx] = reward
      self.next_state_buffer[idx] = next_state
      self.terminal_buffer[idx] = float(1 - terminal)

      self.buffer_counter += 1

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        critic_params_1 = self.critic_1.named_parameters()
        critic_params_2 = self.critic_2.named_parameters()
        target_critic_params_1 = self.target_critic_1.named_parameters()
        target_critic_params_2 = self.target_critic_2.named_parameters()

        critic_state_dict_1 = dict(critic_params_1)
        critic_state_dict_2 = dict(critic_params_2)
        target_critic_dict_1 = dict(target_critic_params_1)
        target_critic_dict_2 = dict(target_critic_params_2)

        for name in critic_state_dict_1:
            critic_state_dict_1[name] = tau*critic_state_dict_1[name].clone() +(1-tau)*target_critic_dict_1[name].clone()
        self.target_critic_1.load_state_dict(critic_state_dict_1)

        for name in critic_state_dict_2:
            critic_state_dict_2[name] = tau*critic_state_dict_2[name].clone() +(1-tau)*target_critic_dict_2[name].clone()
        self.target_critic_2.load_state_dict(critic_state_dict_2)


    def choose_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _, _ = self.actor.sample(state)

        return action.detach().cpu().numpy()[0]


    def update(self):

        if self.buffer_counter > self.batch_size:
            for n in range(self.n_update_iter):

                if self.buffer_counter > self.buffer_size:
                    max_mem = self.buffer_size
                else:
                    max_mem = self.buffer_counter


                batch_indices = np.random.choice(max_mem, self.batch_size)

                state_batch = torch.Tensor(self.state_buffer[batch_indices]).to(self.device)
                action_batch = torch.Tensor(self.action_buffer[batch_indices]).to(self.device)
                reward_batch = torch.Tensor(self.reward_buffer[batch_indices]).to(self.device).reshape((self.batch_size,1))
                next_state_batch = torch.Tensor(self.next_state_buffer[batch_indices]).to(self.device)
                terminal_batch = torch.Tensor(self.terminal_buffer[batch_indices]).to(self.device).reshape((self.batch_size,1))

                next_action_batch,next_logprobs,_ = self.actor.sample(next_state_batch)
                next_action_batch = next_action_batch.detach()
                next_logprobs = next_logprobs.detach()
                next_critic_value_1 = self.target_critic_1(next_state_batch,next_action_batch).detach()
                next_critic_value_2 = self.target_critic_2(next_state_batch,next_action_batch).detach()
                next_critic_value = torch.min(next_critic_value_1,next_critic_value_2) - self.alpha * next_logprobs

                critic_target = reward_batch + terminal_batch * self.gamma * (next_critic_value)

                critic_value_1 = self.critic_1(state_batch, action_batch)
                critic_value_2 = self.critic_2(state_batch, action_batch)
                critic_1_loss = F.mse_loss(critic_value_1, critic_target)
                critic_2_loss = F.mse_loss(critic_value_2, critic_target)

                self.critic_optimizer_1.zero_grad()
                critic_1_loss.backward()
                self.critic_optimizer_1.step()

                self.critic_optimizer_2.zero_grad()
                critic_2_loss.backward()
                self.critic_optimizer_2.step()

                mu,log_probs, _ = self.actor.sample(state_batch)

                critic_val_1 = self.critic_1(state_batch,mu)
                critic_val_2 = self.critic_2(state_batch,mu)
                min_critic_val = torch.min(critic_val_1,critic_val_2)

                actor_loss = ((self.alpha * log_probs) - min_critic_val)
                self.actor_optimizer.zero_grad()
                actor_loss.mean().backward()
                self.actor_optimizer.step()

                if self.tune_alpha:
                    alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()
                    self.alpha = self.log_alpha.exp().detach()

                self.update_network_parameters()
