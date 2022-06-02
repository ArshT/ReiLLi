import sys
import os

dir_list = os.getcwd().split("\\")[:-1]
directory = ""
for word in dir_list:
    directory += word + "\\"
sys.path.append(directory)

from algorithms.SAC.sac_agent import SAC_agent

env_name = "BipedalWalker-v3"
num_episodes = 10
gamma = 0.99
render=True
num_test_episodes = 50
fc1_dims = 512
fc2_dims = 256
device = 'cpu'
solved_reward = 290
batch_size = 512
actor_alpha = 0.0003
critic_alpha = 0.0003
SAC_alpha_lr = 0.0003
SAC_alpha = 0.2
tau = 0.005
buffer_size = 1000000
n_update_iter = 500
tune_alpha = True
model_dir = r"C:\Users\Arsh Tangri\Desktop\Reinforcement Learning\ReiLLi\models"
plot_dir = r"C:\Users\Arsh Tangri\Desktop\Reinforcement Learning\ReiLLi\plots"


sac_agent = SAC_agent(env_name=env_name,num_episodes=num_episodes,batch_size=batch_size,gamma=gamma,solved_reward=solved_reward,tau=tau,buffer_size=buffer_size,
                      render=render,num_test_episodes=num_test_episodes,SAC_alpha=SAC_alpha,n_update_iter=n_update_iter,actor_alpha=actor_alpha,
                      critic_alpha=critic_alpha,SAC_alpha_lr=SAC_alpha_lr,fc1_dims=fc1_dims,fc2_dims=fc2_dims,tune_alpha=tune_alpha,device=device)



sac_agent.train(model_dir=model_dir,plot_dir=plot_dir)
sac_agent.test(model_dir=model_dir)
