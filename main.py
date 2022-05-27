import sys
sys.path.append("C:\\Users\\Arsh Tangri\\Desktop\\Reinforcement Learning\\ReiLLi")
print(sys.path)

from algorithms.A2C.a2c_discrete_agent import A2C_agent

env_name = "LunarLander-v2"
num_episodes = 5000
batch_size = 512
gamma = 0.99
render=True
num_test_episodes = 50
critic_alpha = 0.0001
actor_alpha = 0.0003
fc1_dims = 512
fc2_dims = 256
device = 'cpu'
solved_reward = 200
start_action_std=0.5
action_std_decay_rate=0.05
min_action_std=0.05
action_std_decay_ep=5
buffer_size = 10000000
tau = 0.001
action_std_decay_ep = 3

update_batch_size = 10
alpha = 0.001
K_epochs = 30
eps_clip = 0.2
critic_alpha = 0.0003
actor_alpha = 0.0001
SAC_alpha = 0.2
n_update_iter = 500
SAC_alpha_lr = actor_alpha
tune_alpha = True
target_action_noise_clip = 0.25
action_update_delay = 2





x = A2C_agent(env_name=env_name,num_episodes=num_episodes,update_batch_size=update_batch_size,gamma=gamma,solved_reward=solved_reward,bootstrapping=False,
             render=render,num_test_episodes=num_test_episodes,actor_alpha=actor_alpha,critic_alpha=critic_alpha,fc1_dims=fc1_dims,fc2_dims=fc2_dims,device=device)
x.train()
x.test()
