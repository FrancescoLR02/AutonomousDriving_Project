import numpy as np
import gymnasium
import highway_env
import torch
import random
import os 
import csv

from baseline import BaselineAgent
from Agent import *

np.set_printoptions(linewidth=200, suppress=True, precision=5)


# Set the seed and create the environment
np.random.seed(0)
random.seed(0)
torch.Agent_seed(0)

envName = "highway-v0"
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "normalize": False,   
        "absolute": False,
    },
    'screen_height': 300,
    'screen_width': 1200,
    # "lanes_count": 3,
    # "ego_spacing": 1.5,
    # "policy_frequency": 5,
    # 'duration': 40, 
    'vehicles_count': 50,
    # 'high_speed_reward': 0.8,
    # 'collision_reward': -5,

}

env = gymnasium.make(envName, config=config, render_mode='human')

baseline = False

# Initialize your model and load parameters
if baseline: 
    agent = BaselineAgent(env)
else: 
    agent = Agent(env)
    #checkpoint = torch.load("HighestReward.pth", map_location=torch.device('cpu'))
    checkpoint = torch.load("ppo_highway_agent.pth", map_location=torch.device('cpu'))
    agent.load_state_dict(checkpoint)
    agent.eval()



# Evaluation loop
state, _ = env.reset()
done, truncated = False, False


files = {
    'Data': 'AgentControlActions.csv',
    'Rewards': 'AgentControlRewards.csv'
}
rewardsHeader = ['TotalRewards']
actionsHeader = ['Speed', 'Action', 'Crashed']

needsHeader = {key: not os.path.isfile(path) for key, path in files.items()}


while episode <= 15:
    episodeSteps += 1
    # Select the action to be performed by the agent
    if baseline: action = agent.BasePolicy(state)
    else: 
        state = torch.Tensor(state).flatten()
        action, _, _, _ = agent.GetActionValue(state)

    state, reward, done, truncated, _ = env.step(action)
    env.render()

    episodeReturn += reward


    if done or truncated:
        print(f"Episode Num: {episode} Episode T: {episodeSteps} Return: {episodeReturn:.3f}, Crash: {done}")

        state, _ = env.reset()
        episode += 1
        episodeSteps = 0
        episodeReturn = 0


env.close()
