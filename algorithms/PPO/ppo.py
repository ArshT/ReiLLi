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


class Agent_Discrete():
    def __init__(self,input_dims,n_actions,actor_alpha,critic_alpha,gamma,eps_clip,K_epochs,fc1_dims=256,fc2_dims=256,device='cpu'):
        self.gamma = gamma
        self.critic_loss = nn.MSELoss()
        self.input_dims = input_dims
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.device = torch.device(device)
        self.dist_coeff = 0.01

        self.reward_memory = []
        self.terminal_memory = []
        self.state_memory = torch.empty((1,input_dims))
        self.action_memory = torch.empty((1,)).to(self.device)
        self.log_prob_memory = torch.empty((1,)).to(self.device)

        self.actor_critic = ActorCritic(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,output_dims=n_actions,device=device)

        #self.optimizer = optim.Adam(self.actor_critic.parameters(),lr=self.alpha)
        self.actor_optimizer = optim.Adam(self.actor_critic.actor.parameters(),lr = actor_alpha)
        self.critic_optimizer = optim.Adam(self.actor_critic.critic.parameters(),lr = critic_alpha)
        self.device = torch.device(device)

    def choose_action(self,state):
        action_logits,_ = self.actor_critic.forward(state)
        probabilities = F.softmax(action_logits)
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = (action_probs.log_prob(action)).reshape((1,)).to(self.device)

        action_tensor = action.reshape((1,)).to(self.device)
        self.action_memory = torch.cat((self.action_memory,action_tensor),0)

        self.log_prob_memory = torch.cat((self.log_prob_memory,log_prob),0)

        return action.item()

    def store_transitions(self,reward,state,terminal):
        self.reward_memory.append(reward)

        state = torch.Tensor(state)
        state = state.reshape((1,self.input_dims))
        self.state_memory = torch.cat((self.state_memory,state),0)

        self.terminal_memory.append(terminal)

    def evaluate(self,states,actions):
        action_logits,state_values = self.actor_critic.forward(states)
        probabilities = F.softmax(action_logits).to(self.device)
        action_probs = torch.distributions.Categorical(probabilities)

        logprobs = action_probs.log_prob(actions)
        dist_entropy = action_probs.entropy()

        return logprobs,torch.squeeze(state_values),dist_entropy



    def learn(self,episode_number,batch_size):
        old_logprobs = self.log_prob_memory[1:].to(self.device)
        old_states = self.state_memory[1:]
        old_actions = self.action_memory[1:].to(self.device)

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

        for _ in range(self.K_epochs):
            logprobs,state_values,dist_entropy = self.evaluate(old_states,old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = (G - state_values.detach())

            surr1 = ratios*advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            actor_loss = -torch.min(surr1,surr2)

            critic_loss = self.critic_loss(state_values,G)

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss = actor_loss + 0.5*critic_loss - (0.05/episode_number)*dist_entropy
            loss.mean().backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


        self.reward_memory = []
        self.state_memory = torch.empty((1,self.input_dims))
        self.log_prob_memory = torch.empty((1,)).to(self.device)
        self.action_memory = torch.empty((1,)).to(self.device)
        self.terminal_memory = []



    def learn_multi(self,memory,update_number):
      old_states = memory.states
      old_actions = memory.actions
      old_logprobs = memory.logprobs

      G = memory.disReturn

      for _ in range(self.K_epochs):
        new_logprobs,state_values,dist_entropy = self.evaluate(old_states,old_actions)

        ratios = torch.exp(new_logprobs - old_logprobs.detach())
        advantages = G - state_values.detach()

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios,1-self.eps_clip,1+self.eps_clip) * advantages
        actor_loss = -torch.min(surr1,surr2)

        critic_loss = self.critic_loss(state_values,G)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss = actor_loss + 0.5*critic_loss - (0.05/(update_number*10))*dist_entropy
        loss.mean().backward()
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
