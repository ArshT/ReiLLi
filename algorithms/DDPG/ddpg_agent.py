import numpy as np
import gym
from algorithms.DDPG.ddpg import Agent


class DDPG_agent:
    def __init__(self,env_name,num_episodes,batch_size,gamma,solved_reward,tau,start_action_std,min_action_std,action_std_decay_rate,
                 action_std_decay_ep,buffer_size,render,num_test_episodes,actor_alpha,critic_alpha,fc1_dims,fc2_dims,device):

        self.env = gym.make(env_name)
        input_dims = self.env.observation_space.shape[0]
        action_dims = self.env.action_space.shape[0]

        self.agent = Agent(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,action_dims=action_dims,actor_alpha=actor_alpha,critic_alpha=critic_alpha
                          ,gamma=gamma,start_action_std=start_action_std,min_action_std=min_action_std,action_std_decay_rate=action_std_decay_rate,
                           tau=tau,batch_size=batch_size,buffer_size=buffer_size,device=device)

        self.action_std_decay_ep = action_std_decay_ep

        self.num_episodes = num_episodes
        self.num_test_episodes = num_test_episodes
        self.num_episodes = num_episodes
        self.device = device
        self.render = render
        self.solved_reward = solved_reward
        self.batch_size = batch_size


    def train(self):
        scores = []

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

            if (i+1) % self.action_std_decay_ep == 0:
                self.agent.decay_std()

            if (i+1)%10 == 0:
                avg_score = np.mean(scores[max(0, i-10):(i+1)])
                avg_score_100 = np.mean(scores[max(0, i-100):(i+1)])
                print('episode: ', i+1,'score: ', score,' average_score_10 %.3f' % avg_score,' average_score_100 %.3f' % avg_score_100)

                if avg_score_100 > self.solved_reward:
                    print("Solved!!!!!")
                    break

    def test(self):
        total_score = 0
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
