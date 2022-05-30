import sys
import os

dir_list = os.getcwd().split("\\")[:-1]
directory = ""
for word in dir_list:
    directory += word + "\\"
sys.path.append(directory)

from algorithms.Reinforce.reinforce_discrete_agent import REINFORCE_agent

env_name = "CartPole-v0"
num_episodes = 2000
gamma = 0.99
render=True
num_test_episodes = 50
fc1_dims = 256
fc2_dims = 128
device = 'cpu'
solved_reward = 190
update_batch_size = 10
alpha = 0.0003
model_dir = r"C:\Users\Arsh Tangri\Desktop\Reinforcement Learning\ReiLLi\models"
plot_dir = r"C:\Users\Arsh Tangri\Desktop\Reinforcement Learning\ReiLLi\plots"



reinforce_agent = REINFORCE_agent(env_name=env_name,num_episodes=num_episodes,update_batch_size=update_batch_size,gamma=gamma,solved_reward=solved_reward,
                    render=render,num_test_episodes=num_test_episodes,alpha=alpha,fc1_dims=fc1_dims,fc2_dims=fc2_dims,device=device)


reinforce_agent.train(model_dir=model_dir,plot_dir=plot_dir)
reinforce_agent.test(model_dir=model_dir)
