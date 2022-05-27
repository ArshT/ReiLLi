import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

class DeepQNetwork(nn.Module):
    def __init__(self,input_dims,fc1_dims,fc2_dims,output_dims,device='cpu'):
        super(DeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims,fc1_dims)
        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        self.out = nn.Linear(fc2_dims,output_dims)

        self.device = torch.device(device)
        self.to(self.device)

    def forward(self,observation):
        state = torch.Tensor(observation).to(self.device)

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        Q_values = self.out(x)

        return Q_values


class Agent():
    def __init__(self,input_dims,n_actions,alpha,gamma,batch_size,replace,epsilon=1.0,fc1_dims=256,
                 fc2_dims=256,buffer_size=100000,eps_dec=0.9999,eps_end=0.01,device='cpu'):

        self.alpha = alpha
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_end = eps_end
        self.gamma = gamma
        self.batch_size = batch_size
        self.replace_limit = replace
        self.device = torch.device(device)
        self.buffer_size = buffer_size
        self.n_actions = n_actions

        self.action_space = [i for i in range(n_actions)]
        self.learn_counter = 0
        self.buffer_counter = 0

        self.Q_eval = DeepQNetwork(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,output_dims=n_actions,device=device)
        self.Q_target = DeepQNetwork(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,output_dims=n_actions,device=device)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

        self.optimizer = optim.Adam(self.Q_eval.parameters(),lr=self.alpha)
        self.loss = nn.MSELoss()

        self.state_buffer = np.zeros((self.buffer_size,input_dims))
        self.action_buffer = np.zeros((self.buffer_size,1))
        self.reward_buffer = np.zeros((self.buffer_size,1))
        self.next_state_buffer = np.zeros((self.buffer_size,input_dims))
        self.terminal_buffer = np.zeros((self.buffer_size,1))

    def store_transitions(self,state,action,reward,next_state,terminal):
        index = self.buffer_counter % self.buffer_size

        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_state
        self.terminal_buffer[index] = 1 - terminal

        self.buffer_counter += 1

    def choose_action(self,state):
        rand = np.random.random()

        if rand > self.epsilon:
            q_values = self.Q_eval.forward(state)
            action = torch.argmax(q_values).item()
        else:
            action = random.choice(self.action_space)

        return action

    def learn(self):
        if self.learn_counter % self.replace_limit == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

        if self.buffer_counter >= self.batch_size:
            self.optimizer.zero_grad()

            if self.buffer_counter > self.buffer_size:
                max_mem = self.buffer_size
            else:
                max_mem = self.buffer_counter

            batch_indices = np.random.choice(max_mem, self.batch_size)
            state_batch = self.state_buffer[batch_indices]

            action_indices = self.action_buffer[batch_indices].reshape((self.batch_size,))
            action_indices = action_indices.astype('int32')

            reward_batch = self.reward_buffer[batch_indices]
            next_state_batch = self.next_state_buffer[batch_indices]
            terminal_batch = self.terminal_buffer[batch_indices]

            reward_batch = torch.Tensor(reward_batch).to(self.device)
            terminal_batch = torch.Tensor(terminal_batch).to(self.device)

            q_eval = self.Q_eval.forward(state_batch)
            q_target = q_eval.clone()
            q_next = self.Q_target.forward(next_state_batch).detach()

            indices = np.arange(self.batch_size, dtype=np.int32)
            q_next_max = (torch.max(q_next,dim=1)[0]).reshape((self.batch_size,1))
            q_target[indices,action_indices] = (reward_batch + self.gamma * q_next_max * terminal_batch).reshape((self.batch_size,))

            loss = self.loss(q_eval,q_target)
            loss.backward()
            self.optimizer.step()
            self.learn_counter += 1

            if self.epsilon > self.eps_end:
                self.epsilon *= self.eps_dec
            else:
                self.epsilon = self.eps_end
