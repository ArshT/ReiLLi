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
    def __init__(self,input_dims,n_actions,alpha,gamma,epsilon,batch_size,replace,fc1_dims=256,
                 fc2_dims=256,eps_dec=0.9999,eps_end=0.01,buffer_size=100000,device='cpu'):

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.buffer_size = buffer_size
        self.action_space = [i for i in range(self.n_actions)]
        self.device = torch.device(device)
        self.replace_limit = replace

        self.buffer_counter = 0
        self.learn_counter = 0

        self.Q_eval = DeepQNetwork(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,output_dims=n_actions,device=device)
        self.Q_target = DeepQNetwork(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,output_dims=n_actions,device=device)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.Q_eval.parameters(),lr=self.alpha)

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

        if self.buffer_counter > self.batch_size:
            self.optimizer.zero_grad()

            if self.buffer_counter < self.buffer_size:
                max_mem = self.buffer_counter
            else:
                max_mem = self.buffer_size

            batch_indices = np.random.choice(max_mem, self.batch_size)
            state_batch = self.state_buffer[batch_indices]

            action_indices = self.action_buffer[batch_indices].reshape((self.batch_size,))
            action_indices = action_indices.astype('int32')

            reward_batch = self.reward_buffer[batch_indices]
            next_state_batch = self.next_state_buffer[batch_indices]
            terminal_batch = self.terminal_buffer[batch_indices]

            reward_batch =  torch.Tensor(reward_batch).to(self.device)
            terminal_batch = torch.Tensor(terminal_batch).to(self.device)

            q_eval = self.Q_eval.forward(state_batch)
            q_target = q_eval.clone()

            q_next_target = self.Q_target.forward(next_state_batch)
            q_next_eval = self.Q_eval.forward(next_state_batch)
            next_action_indices = torch.argmax(q_next_eval,dim=1)

            indices = np.arange(self.batch_size, dtype=np.int32)
            q_next = (q_next_target[indices,next_action_indices]).reshape((self.batch_size,1))

            q_target[indices,action_indices] = (reward_batch + self.gamma * q_next * terminal_batch).reshape((self.batch_size,))

            loss = self.loss(q_eval,q_target)
            loss.backward()
            self.optimizer.step()

            self.learn_counter += 1

            if self.epsilon > self.eps_end:
                self.epsilon *= self.eps_dec
            else:
                self.epsilon = self.eps_end
