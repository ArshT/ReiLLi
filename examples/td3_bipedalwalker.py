import sys
import os

dir_list = os.getcwd().split("\\")[:-1]
directory = ""
for word in dir_list:
    directory += word + "\\"
sys.path.append(directory)

from algorithms.TD3.td3_agent import TD3_agent

env_name = "BipedalWalker-v3"
num_episodes = 10
gamma = 0.99
render=True
num_test_episodes = 50
fc1_dims = 400
fc2_dims = 300
device = 'cpu'
solved_reward = 290
batch_size = 512
actor_alpha = 0.00005
critic_alpha = 0.0001
start_action_std = 0.5
action_std_decay_rate = 0.05
min_action_std = 0.05
action_std_decay_ep = 130
tau = 0.001
buffer_size = 1000000
target_action_noise_clip = 0.25
n_update_iter = 500
action_update_delay = 2
model_dir = r"C:\Users\Arsh Tangri\Desktop\Reinforcement Learning\ReiLLi\models"
plot_dir = r"C:\Users\Arsh Tangri\Desktop\Reinforcement Learning\ReiLLi\plots"


td3_agent = TD3_agent(env_name=env_name,num_episodes=num_episodes,batch_size=batch_size,gamma=gamma,solved_reward=solved_reward,tau=tau,start_action_std=start_action_std,
                      min_action_std=min_action_std,action_std_decay_rate=action_std_decay_rate,action_std_decay_ep=action_std_decay_ep,buffer_size=buffer_size,
                      render=render,num_test_episodes=num_test_episodes,actor_alpha=actor_alpha,critic_alpha=critic_alpha,fc1_dims=fc1_dims,fc2_dims=fc2_dims,device=device,
                      target_action_noise_clip=target_action_noise_clip,n_update_iter=n_update_iter,action_update_delay=action_update_delay)



td3_agent.train(model_dir=model_dir,plot_dir=plot_dir)
td3_agent.test(model_dir=model_dir)
