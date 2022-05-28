import sys
import os

dir_list = os.getcwd().split("\\")[:-1]
directory = ""
for word in dir_list:
    directory += word + "\\"
sys.path.append(directory)

from algorithms.DDPG.ddpg_agent import DDPG_agent

env_name = "BipedalWalker-v3"
num_episodes = 5000
gamma = 0.99
render=True
num_test_episodes = 50
fc1_dims = 512
fc2_dims = 256
device = 'cpu'
solved_reward = 285
batch_size = 512
actor_alpha = 0.000001
critic_alpha = 0.000005
start_action_std = 0.5
action_std_decay_rate = 0.05
min_action_std = 0.05
action_std_decay_ep = 200
tau = 0.001
buffer_size = 1000000


ddpg_agent = DDPG_agent(env_name=env_name,num_episodes=num_episodes,batch_size=batch_size,gamma=gamma,solved_reward=solved_reward,tau=tau,start_action_std=start_action_std,
                      min_action_std=min_action_std,action_std_decay_rate=action_std_decay_rate,action_std_decay_ep=action_std_decay_ep,buffer_size=buffer_size,
                      render=render,num_test_episodes=num_test_episodes,actor_alpha=actor_alpha,critic_alpha=critic_alpha,fc1_dims=fc1_dims,fc2_dims=fc2_dims,device=device)



ddpg_agent.train()
ddpg_agent.test()
