# ReiLLi

ReiLLi is a PyTorch project that consists of easy-to-understand and reliable implementations of several Model-Free Reinforcement Learning algorithms. The project is primarily intended for beginners to help them understand and conveniently use Model-Free RL algorithms  for training agents  that can perform optimally in several popular OpenAI gym environments

## Algorithms
The following algorithms are implemented in this project:
- Deep Q-Learning with discrete action space
- Double Deep Q-Learning with discrete action space
- REINFORCE with discrete action space
- A2C with discrete action space
- PPO with both discrete and continuous action space
- DDPG with continuous action space
- TD3 with discretized action space
- SAC with continuous action space


## Usage
- The 'examples' folder contains example files for every algorithm. These examples can be followed to use any algorithm for training agents for environments of one's choice.
- The train function is used for training the agent and the test function is used for testing the trained agent. The test function can also be used to run pre-trained agents using the dict files stored in the 'models' folder.
