import numpy as np
import matplotlib.pyplot as plt
import os


class RewardPlot:
    def __init__(self,env_name,algo_name,save_dir):
        self.plot_title_name = algo_name + "_" + env_name
        self.save_dir = save_dir


    def plot_reward_curve(self,episode_reward_list):
        rewards = np.array(episode_reward_list)
        indices = np.arange(len(episode_reward_list))

        plt.plot(indices,rewards)
        plt.title(self.plot_title_name)
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt_dir = os.path.join(self.save_dir,self.plot_title_name+".png")
        plt.savefig(plt_dir,bbox_inches='tight')
