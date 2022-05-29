import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np




class PolicyNetwork(nn.Module):
    def __init__(self,input_dims,fc1_dims,fc2_dims,output_dims,device='cpu'):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims,fc1_dims)
        self.fc2 = nn.Linear(fc1_dims,fc2_dims)
        self.out = nn.Linear(fc2_dims,output_dims)

        self.device = torch.device(device)
        self.to(self.device)

    def forward(self,observation):
        state = torch.Tensor(observation).to(self.device)

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.out(x)

        return action_logits


class Agent():
    def __init__(self,alpha,gamma,input_dims,n_actions,fc1_dims=256,fc2_dims=256,device='cpu'):

        self.device = torch.device(device)
        self.reward_memory = []
        self.log_prob_memory = torch.empty((1,)).to(self.device)
        self.terminal_memory = []
        self.input_dims = input_dims

        self.alpha = alpha
        self.gamma = gamma

        self.policy = PolicyNetwork(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,output_dims=n_actions,device=device)

        self.optimizer = optim.Adam(self.policy.parameters(),lr=self.alpha)

        self.state_memory = torch.empty((1,input_dims))
        self.action_memory = torch.empty((1,)).to(self.device)

        self.input_dims = input_dims

    def choose_action(self,state):

        probabilities = F.softmax(self.policy.forward(state))
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()

        log_prob = action_probs.log_prob(action).reshape((1,))
        self.log_prob_memory = torch.cat((self.log_prob_memory,log_prob),0)

        action_tensor = action.reshape((1,)).to(self.device)
        self.action_memory = torch.cat((self.action_memory,action_tensor),0)

        return action.item()


    def store_rewards(self,reward,terminal):
        self.reward_memory.append(reward)
        self.terminal_memory.append(terminal)

    def learn(self,batch_size):

        self.log_prob_memory = self.log_prob_memory[1:]

        G = []
        G_sum = 0

        for reward,done in zip(reversed(self.reward_memory),reversed(self.terminal_memory)):
            if done:
                G_sum = 0
            G_sum = reward + self.gamma * G_sum
            G.append(G_sum)

        G = G[::-1]
        G = np.array(G)
        G = (torch.Tensor(G)).to(self.device)

        loss = torch.sum(-self.log_prob_memory * G)

        self.optimizer.zero_grad()
        loss /= batch_size
        loss.backward()
        self.optimizer.step()

        self.reward_memory = []
        self.log_prob_memory = torch.empty((1,)).to(self.device)
        self.terminal_memory = []
        self.state_memory = torch.empty((1,self.input_dims))
        self.action_memory = torch.empty((1,)).to(self.device)


    def update_multi(self,memory,num_agents):
      log_prob_memory = memory.logprobs
      G = memory.disReturn
      states = memory.states
      actions = memory.actions

      probabilities = F.softmax(self.policy.forward(states))
      action_probs = torch.distributions.Categorical(probabilities)
      new_log_prob_memory = (action_probs.log_prob(actions)).to(self.device)


      self.optimizer.zero_grad()
      loss = torch.sum(-new_log_prob_memory * G) / num_agents
      loss.backward()
      self.optimizer.step()


    def store_transitions_multi(self,state,reward,terminal):
        self.reward_memory.append(reward)

        state = torch.Tensor(state)
        state = state.reshape((1,self.input_dims))
        self.state_memory = torch.cat((self.state_memory,state),0)

        self.terminal_memory.append(terminal)


    def experience_to_tensor(self,update_timestep):
      state_tensor = self.state_memory[1:update_timestep+1]
      action_tensor = self.action_memory[1:update_timestep+1]
      logprob_tensor = self.log_prob_memory[1:update_timestep+1]

      G = []
      G_sum = 0
      for reward,terminal in zip(reversed(self.reward_memory),reversed(self.terminal_memory)):
        if terminal:
          G_sum = 0
        G_sum = reward + self.gamma * G_sum
        G.append(G_sum)

      G = G[::-1]
      G = np.array(G)
      G = torch.Tensor(G).to(self.device)
      G_mean = torch.mean(G)
      G_std = torch.std(G)
      G_tensor = (G - G_mean)/(G_std + 1e-5)
      G_tensor = G_tensor[:update_timestep]

      return state_tensor,action_tensor,logprob_tensor,G_tensor
