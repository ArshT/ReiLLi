import sys
import os

dir_list = os.getcwd().split("\\")[:-1]
directory = ""
for word in dir_list:
    directory += word + "\\"
sys.path.append(directory)



from algorithms.A2C.a2c_discrete_multi_agent import A2C_multi_agent



if __name__ == '__main__':
    env_name = "LunarLander-v2"
    num_agents = 10
    max_timestep_per_ep = 500
    update_timestep = 2500
    solved_reward = 200
    max_rounds = 500

    fc1_dims = 256
    fc2_dims = 128
    actor_alpha = 0.0001
    critic_alpha = 0.0003
    gamma = 0.99
    K_epochs = 50
    eps_clip = 0.2
    continuous = False

    a2c_multi_agent = A2C_multi_agent(env_name=env_name,num_agents=num_agents,max_rounds=max_rounds,max_timesteps_per_ep=max_timestep_per_ep,
                                  update_timestep=update_timestep,solved_reward=solved_reward,fc1_dims=fc1_dims,fc2_dims=fc2_dims,actor_alpha=actor_alpha,gamma=gamma,
                                  critic_alpha=critic_alpha)

    a2c_multi_agent.train()
    a2c_multi_agent.test()
