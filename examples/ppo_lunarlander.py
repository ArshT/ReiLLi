import sys
import os

dir_list = os.getcwd().split("\\")[:-1]
directory = ""
for word in dir_list:
    directory += word + "\\"
sys.path.append(directory)

from algorithms.PPO.ppo_agent import PPO_agent

env_name = "LunarLander-v2"
num_episodes = 5000
gamma = 0.99
render=True
num_test_episodes = 50
fc1_dims = 256
fc2_dims = 128
device = 'cpu'
solved_reward = 200
update_batch_size = 10
actor_alpha = 0.0001
critic_alpha = 0.0003
eps_clip = 0.2
K_epochs = 30


ppo_agent = PPO_agent(env_name=env_name,continuous=False,num_episodes=num_episodes,update_batch_size=update_batch_size,gamma=gamma,solved_reward=solved_reward,actor_alpha=actor_alpha,critic_alpha=critic_alpha,
                                 eps_clip=eps_clip,K_epochs=K_epochs,render=render,num_test_episodes=num_test_episodes,fc1_dims=fc1_dims,fc2_dims=fc2_dims,device=device)


ppo_agent.train()
ppo_agent.test()
