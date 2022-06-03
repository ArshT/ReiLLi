import sys
import os

dir_list = os.getcwd().split("\\")[:-1]
directory = ""
for word in dir_list:
    directory += word + "\\"
sys.path.append(directory)



from algorithms.Reinforce.reinforce_discrete_multi_agent import Reinforce_multi_agent



if __name__ == '__main__':
    env_name = "LunarLander-v2"
    num_agents = 10
    max_timestep_per_ep = 500
    update_timestep = 2500
    solved_reward = 200
    max_rounds = 500

    fc1_dims = 256
    fc2_dims = 128
    alpha = 0.001
    gamma = 0.99
    num_test_episodes = 50
    render = True

    model_dir = r"C:\Users\Arsh Tangri\Desktop\Reinforcement Learning\ReiLLi\models"
    plot_dir = r"C:\Users\Arsh Tangri\Desktop\Reinforcement Learning\ReiLLi\plots"


    reinforce_multi_agent = Reinforce_multi_agent(env_name=env_name,num_agents=num_agents,max_rounds=max_rounds,max_timesteps_per_ep=max_timestep_per_ep,
                                              update_timestep=update_timestep,solved_reward=solved_reward,fc1_dims=fc1_dims,fc2_dims=fc2_dims,alpha=alpha,gamma=gamma,
                                              num_test_episodes=num_test_episodes,render = render)

    reinforce_multi_agent.train()
    reinforce_multi_agent.test()
