import numpy as np
import gym
from algorithms.PPO.ppo import Agent_Discrete
from algorithms.PPO.ppo_continuous import Agent_Continuous
from utils.plot import RewardPlot
import os
import torch



class PPO_agent:
    def __init__(self,env_name,continuous,num_episodes,update_batch_size,gamma,solved_reward,actor_alpha,critic_alpha,
                 eps_clip,K_epochs,render,num_test_episodes,fc1_dims,fc2_dims,device,
                 start_action_std=None,action_std_decay_rate=None,min_action_std=None,action_std_decay_ep=None):

        self.env = gym.make(env_name)
        self.env_name = env_name
        self.continuous = continuous

        if continuous:
            input_dims = self.env.observation_space.shape[0]
            action_dims = self.env.action_space.shape[0]

            self.agent = Agent_Continuous(gamma=gamma,actor_alpha=actor_alpha,critic_alpha=critic_alpha,input_dims=input_dims,K_epochs=K_epochs,
                               eps_clip=eps_clip,action_dims=action_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,device=device,
                               start_action_std=start_action_std,action_std_decay_rate=action_std_decay_rate,min_action_std=min_action_std)
        else:
            input_dims = self.env.observation_space.shape[0]
            n_actions = self.env.action_space.n
            self.agent = Agent_Discrete(gamma=gamma,actor_alpha=actor_alpha,critic_alpha=critic_alpha,input_dims=input_dims,K_epochs=K_epochs,
                           eps_clip=eps_clip,n_actions=n_actions,fc1_dims=fc1_dims,fc2_dims=fc2_dims,device=device)

        self.num_episodes = num_episodes
        self.num_test_episodes = num_test_episodes
        self.num_episodes = num_episodes
        self.device = device
        self.render = render
        self.update_batch_size = update_batch_size
        self.solved_reward = solved_reward
        self.action_std_decay_ep = action_std_decay_ep


    def train(self,model_dir = None,plot_dir=None):
        scores = []

        if plot_dir:
            if self.continuous:
                plot_graph = RewardPlot(env_name=self.env_name,algo_name="PPO_Continuous",save_dir=plot_dir)
            else:
                plot_graph = RewardPlot(env_name=self.env_name,algo_name="PPO",save_dir=plot_dir)

        for i in range(self.num_episodes):
            done = False
            score = 0
            observation = self.env.reset()
            while not done:
                action = self.agent.choose_action(observation)
                observation_,reward,done,_ = self.env.step(action)
                self.agent.store_transitions(reward,observation,done)
                observation = observation_
                score += reward
            scores.append(score)

            if self.action_std_decay_ep:
                if (i+1)%self.action_std_decay_ep == 0:
                    self.agent.decay_std()


            if (i+1)%self.update_batch_size == 0:
                self.agent.learn(i+1,self.update_batch_size)

                avg_score = np.mean(scores[max(0, i-10):(i+1)])
                avg_score_100 = np.mean(scores[max(0, i-100):(i+1)])
                print('episode: ', i+1,'score: ', score,' average_score_10 %.3f' % avg_score,' average_score_100 %.3f' % avg_score_100)
                print()

                if not plot_dir:
                    if avg_score_100 > self.solved_reward:
                        print("Solved!!!!")
                        break

        if model_dir:
            if self.continuous:
                save_dir = os.path.join(model_dir,"ppo_continuous_dict_"+self.env_name+".pth")
            else:
                save_dir = os.path.join(model_dir,"ppo_dict_"+self.env_name+".pth")

            torch.save(self.agent.actor_critic.state_dict(),save_dir)


        if plot_dir:
            plot_graph.plot_reward_curve(episode_reward_list=scores)


    def test(self,model_dir = None):
        total_score = 0

        if model_dir:
            if self.continuous:
                save_dir = os.path.join(model_dir,"ppo_continuous_dict_"+self.env_name+".pth")
            else:
                save_dir = os.path.join(model_dir,"ppo_dict_"+self.env_name+".pth")

            self.agent.actor_critic.load_state_dict(torch.load(save_dir))

        for i in range(self.num_test_episodes):
            done = False
            observation = self.env.reset()
            score = 0
            while not done:
                action = self.agent.choose_action(observation)
                observation_, reward, done, info = self.env.step(action)
                observation = observation_
                score += reward
                if self.render:
                    self.env.render()

            total_score += score

        print("Average Score:", total_score/self.num_test_episodes)
