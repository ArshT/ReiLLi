import numpy as np
import gym
from algorithms.A2C.a2c_discrete import Agent
import os
import torch
from utils.plot import RewardPlot


class A2C_agent:
    def __init__(self,env_name,num_episodes,update_batch_size,gamma,solved_reward,bootstrapping,
                 render,num_test_episodes,actor_alpha,critic_alpha,fc1_dims,fc2_dims,device):

        self.env = gym.make(env_name)
        input_dims = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        self.env_name = env_name

        self.agent = Agent(input_dims=input_dims,n_actions=n_actions,actor_alpha=actor_alpha,critic_alpha=critic_alpha,gamma=gamma,
                           fc1_dims=fc1_dims,fc2_dims=fc2_dims,device=device,bootstrapping=bootstrapping)

        self.num_episodes = num_episodes
        self.num_test_episodes = num_test_episodes
        self.num_episodes = num_episodes
        self.device = device
        self.render = render
        self.update_batch_size = update_batch_size
        self.solved_reward = solved_reward

    def train(self,model_dir = None,plot_dir=None):
        scores = []

        if plot_dir:
            plot_graph = RewardPlot(env_name=self.env_name,algo_name="A2C",save_dir=plot_dir)

        for i in range(self.num_episodes):
            done = False
            score = 0
            observation = self.env.reset()
            while not done:
                action = self.agent.choose_action(observation)
                observation_,reward,done,_ = self.env.step(action)
                self.agent.store_transitions(reward,observation,done,observation_)
                observation = observation_
                score += reward
            scores.append(score)

            if (i+1)%self.update_batch_size == 0:
                self.agent.learn()

                avg_score = np.mean(scores[max(0, i-10):(i+1)])
                avg_score_100 = np.mean(scores[max(0, i-100):(i+1)])
                print('episode: ', i+1,'score: ', score,' average_score_10 %.3f' % avg_score,' average_score_100 %.3f' % avg_score_100)
                print()

                if not plot_dir:
                    if avg_score_100 > self.solved_reward:
                        print("Solved!!!!")
                        break


        if model_dir:
            save_dir = os.path.join(model_dir,"a2c_dict_"+self.env_name+".pth")
            torch.save(self.agent.actor_critic.state_dict(),save_dir)


        if plot_dir:
            plot_graph.plot_reward_curve(episode_reward_list=scores)

    def test(self,model_dir=None):
        total_score = 0

        if model_dir:
            save_dir = os.path.join(model_dir,"a2c_dict_"+self.env_name+".pth")
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
