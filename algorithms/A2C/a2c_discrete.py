import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self,input_dims,fc1_dims,fc2_dims,output_dims,device='cpu'):
        super(Actor, self).__init__()

        self.fc1_actor = nn.Linear(input_dims,fc1_dims)
        self.fc2_actor = nn.Linear(fc1_dims,fc2_dims)
        self.out_actor = nn.Linear(fc2_dims,output_dims)

        self.device = torch.device(device)
        self.to(self.device)

    def forward(self,observation):
        state = torch.Tensor(observation).to(self.device)

        x = F.relu(self.fc1_actor(state))
        x = F.relu(self.fc2_actor(x))
        action_logits = self.out_actor(x)

        return action_logits


class Critic(nn.Module):
    def __init__(self,input_dims,fc1_dims,fc2_dims,output_dims,device='cpu'):
        super(Critic, self).__init__()

        self.fc1_critic = nn.Linear(input_dims,fc1_dims)
        self.fc2_critic = nn.Linear(fc1_dims,fc2_dims)
        self.out_critic = nn.Linear(fc2_dims,1)

        self.device = torch.device(device)
        self.to(self.device)

    def forward(self,observation):
        state = torch.Tensor(observation).to(self.device)

        x = F.relu(self.fc1_critic(state))
        x = F.relu(self.fc2_critic(x))
        value = self.out_critic(x)

        return value

class ActorCritic(nn.Module):
    def __init__(self,input_dims,fc1_dims,fc2_dims,output_dims,device='cpu'):
        super(ActorCritic, self).__init__()

        self.actor = Actor(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,output_dims=output_dims,device=device)
        self.critic = Critic(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,output_dims=output_dims,device=device)

        self.device = torch.device(device)
        self.to(self.device)

    def forward(self,state):

        action_logits = self.actor(state)
        value = self.critic(state)

        return action_logits,value


class Agent():
    def __init__(self,input_dims,n_actions,actor_alpha,critic_alpha,gamma,fc1_dims=256,fc2_dims=256,device='cpu',bootstrapping='False'):
        self.gamma = gamma
        self.input_dims = input_dims
        self.bootstrapping = bootstrapping
        self.device = torch.device(device)

        self.reward_memory = []
        self.terminal_memory = []
        self.state_memory = torch.empty((1,input_dims))
        self.log_prob_memory = torch.empty((1,)).to(self.device)
        self.next_state_memory = torch.empty((1,input_dims))
        self.action_memory = torch.empty((1,)).to(self.device)

        self.actor_critic = ActorCritic(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,output_dims=n_actions,device=device)

        self.actor_optimizer = optim.Adam(self.actor_critic.actor.parameters(),lr = actor_alpha)
        self.critic_optimizer = optim.Adam(self.actor_critic.critic.parameters(),lr = critic_alpha)
        self.critic_loss = nn.MSELoss()

    def choose_action(self,state):
        action_logits,_ = self.actor_critic.forward(state)
        probabilities = F.softmax(action_logits)
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()

        log_prob = (action_probs.log_prob(action)).reshape((1,)).to(self.device)
        self.log_prob_memory = torch.cat((self.log_prob_memory,log_prob),0)

        action_tensor = action.reshape((1,)).to(self.device)
        self.action_memory = torch.cat((self.action_memory,action_tensor),0)

        return action.item()

    def store_transitions(self,reward,state,terminal,next_state):
        self.reward_memory.append(reward)

        state = torch.Tensor(state)
        state = state.reshape((1,self.input_dims))
        self.state_memory = torch.cat((self.state_memory,state),0)

        self.terminal_memory.append(terminal)

        next_state = torch.Tensor(next_state)
        next_state = next_state.reshape((1,self.input_dims))
        self.next_state_memory = torch.cat((self.next_state_memory,state),0)

    def get_dist_entropy(self,states):
        action_logits,_ = self.actor_critic.forward(states)
        probabilities = F.softmax(action_logits)
        action_probs = torch.distributions.Categorical(probabilities)
        dist_entropy = action_probs.entropy()

        return dist_entropy

    def learn(self):
        self.log_prob_memory = self.log_prob_memory[1:]
        self.state_memory = self.state_memory[1:]
        self.next_state_memory = self.next_state_memory[1:]

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
        G = (G - G_mean)/(G_std + 1e-5)

        _,values = self.actor_critic.forward(self.state_memory)
        values = torch.squeeze(values)
        advantages = (G - values.detach())
        dist_entropy = self.get_dist_entropy(self.state_memory)

        actor_loss = torch.sum(-self.log_prob_memory*advantages)
        critic_loss = self.critic_loss(values,G)


        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss = actor_loss + 0.5*critic_loss
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        self.reward_memory = []
        self.state_memory = torch.empty((1,self.input_dims))
        self.log_prob_memory = torch.empty((1,)).to(self.device)
        self.next_state_memory = torch.empty((1,self.input_dims))
        self.terminal_memory = []
        self.action_memory = torch.empty((1,)).to(self.device)


    def update_multi(self,memory,num_agents):
      G = memory.disReturn
      states = memory.states
      actions = memory.actions

      action_logits,values = self.actor_critic.forward(states)
      values = torch.squeeze(values)
      advantages = (G - values.detach())

      probabilities = F.softmax(action_logits)
      action_probs = torch.distributions.Categorical(probabilities)
      log_prob_memory = (action_probs.log_prob(actions)).to(self.device)

      actor_loss = torch.sum(-log_prob_memory*advantages)
      critic_loss = self.critic_loss(values,G)

      self.actor_optimizer.zero_grad()
      self.critic_optimizer.zero_grad()
      loss = actor_loss + 0.5*critic_loss
      loss.backward()
      self.actor_optimizer.step()
      self.critic_optimizer.step()


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
