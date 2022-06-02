import numpy as np
import gym
from algorithms.SAC.sac import Agent
import os
import torch
from utils.plot import RewardPlot

class SAC_agent:
    def __init__(self,env_name,num_episodes,batch_size,gamma,solved_reward,tau,buffer_size,render,num_test_episodes,SAC_alpha,n_update_iter,
                 actor_alpha,critic_alpha,SAC_alpha_lr,fc1_dims,fc2_dims,tune_alpha,device):

        self.env = gym.make(env_name)
        input_dims = self.env.observation_space.shape[0]
        action_dims = self.env.action_space.shape[0]
        self.env_name = env_name

        self.agent = Agent(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,action_dims=action_dims,gamma=gamma,tau=tau,
                           SAC_alpha=SAC_alpha,actor_alpha=actor_alpha,critic_alpha=critic_alpha,SAC_alpha_lr=SAC_alpha_lr,
                           batch_size=batch_size,buffer_size=buffer_size,tune_alpha=tune_alpha,n_update_iter=n_update_iter,device=device)

        self.num_episodes = num_episodes
        self.num_test_episodes = num_test_episodes
        self.num_episodes = num_episodes
        self.device = device
        self.render = render
        self.solved_reward = solved_reward
        self.batch_size = batch_size
        self.n_update_iter = n_update_iter


    def train(self,model_dir=None,plot_dir=None):
        scores = []

        if plot_dir:
            plot_graph = RewardPlot(env_name=self.env_name,algo_name="SAC",save_dir=plot_dir)

        for i in range(self.num_episodes):
            done = False
            score = 0
            observation = self.env.reset()

            while not done:
                action = self.agent.choose_action(observation)
                observation_,reward,done,_ = self.env.step(action)
                self.agent.store_transitions(observation,action,reward,observation_,done)
                score += reward
                observation = observation_

            self.agent.update()
            scores.append(score)

            if (i+1)%10 == 0:
                avg_score = np.mean(scores[max(0, i-10):(i+1)])
                avg_score_100 = np.mean(scores[max(0, i-100):(i+1)])
                print('episode: ', i+1,'score: ', score,' average_score_10 %.3f' % avg_score,' average_score_100 %.3f' % avg_score_100)
                print()

                if not plot_dir:
                    if avg_score_100 > self.solved_reward:
                        print("Solved!!!!")
                        breakk

        if model_dir:
            save_dir = os.path.join(model_dir,"sac_dict_"+self.env_name+".pth")
            torch.save(self.agent.actor.state_dict(),save_dir)

        if plot_dir:
            plot_graph.plot_reward_curve(episode_reward_list=scores)


    def test(self,model_dir=None):
        total_score = 0

        if model_dir:
            save_dir = os.path.join(model_dir,"sac_dict_"+self.env_name+".pth")
            self.agent.actor.load_state_dict(torch.load(save_dir))

        for i in range(self.num_test_episodes):
            done = False
            observation = self.env.reset()
            score = 0
            self.agent.epsilon = 0.0
            while not done:
                action = self.agent.choose_action(observation)
                observation_, reward, done, info = self.env.step(action)
                observation = observation_
                score += reward
                if self.render:
                    self.env.render()

            total_score += score

        print("Average Score:", total_score/self.num_test_episodes)
