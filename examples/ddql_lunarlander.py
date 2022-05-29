import sys
import os

dir_list = os.getcwd().split("\\")[:-1]
directory = ""
for word in dir_list:
    directory += word + "\\"
sys.path.append(directory)

from algorithms.Double_Deep_Q_Learning.double_deep_q_learning_agent import Double_DQL_agent

env_name = "LunarLander-v2"
num_episodes = 5000
gamma = 0.99
render=True
num_test_episodes = 50
fc1_dims = 256
fc2_dims = 128
device = 'cpu'
solved_reward = 215
batch_size = 256
alpha = 0.0003
replace = 64
epsilon = 1.0
eps_dec = 0.9999
eps_end = 0.001
buffer_size = 1000000
model_dir = r"C:\Users\Arsh Tangri\Desktop\Reinforcement Learning\ReiLLi\models"
plot_dir = r"C:\Users\Arsh Tangri\Desktop\Reinforcement Learning\ReiLLi\plots"

ddql_agent = Double_DQL_agent(env_name=env_name,num_episodes=num_episodes,batch_size=batch_size,gamma=gamma,solved_reward=solved_reward,replace=replace,epsilon=epsilon,
                      buffer_size=buffer_size,render=render,num_test_episodes=num_test_episodes,alpha=alpha,eps_dec=eps_dec,eps_end=eps_end,fc1_dims=fc1_dims,fc2_dims=fc2_dims,device=device)



ddql_agent.train(model_dir=model_dir,plot_dir=plot_dir)
ddql_agent.test(model_dir=model_dir)
