import sys
import os

dir_list = os.getcwd().split("\\")[:-1]
directory = ""
for word in dir_list:
    directory += word + "\\"
sys.path.append(directory)

from algorithms.PPO.ppo_multi_agent import PPO_multi_agent

if __name__ == '__main__':
    env_name = "LunarLanderContinuous-v2"
    num_agents = 10
    max_timestep_per_ep = 500
    update_timestep = 2500
    solved_reward = 200
    max_rounds = 500

    fc1_dims = 256
    fc2_dims = 256
    actor_lr = 0.0003
    critic_lr = 0.0003
    gamma = 0.99
    K_epochs = 50
    eps_clip = 0.2
    continuous = True
    start_action_std = 0.5
    action_std_decay_rate = 0.05
    min_action_std = 0.05
    action_std_decay_round = 10
    render = True
    num_test_episodes = 50

    

    ppo_multi_agent = PPO_multi_agent(env_name=env_name,num_agents=num_agents,max_rounds=max_rounds,max_timesteps_per_ep=max_timestep_per_ep,
                                  update_timestep=update_timestep,solved_reward=solved_reward,fc1_dims=fc1_dims,fc2_dims=fc2_dims,actor_alpha=actor_lr,
                                  critic_alpha=critic_lr,gamma=gamma,K_epochs=K_epochs,eps_clip=eps_clip,continuous=continuous,start_action_std=start_action_std,
                                  action_std_decay_rate=action_std_decay_rate,min_action_std=min_action_std,action_std_decay_round=action_std_decay_round,num_test_episodes=num_test_episodes,
                                  render=render)

    ppo_multi_agent.train()
    ppo.PPO_multi_agent.test()
