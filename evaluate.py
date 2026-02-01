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
#torch.Agent_seed(0)

baseline = False


envName = "highway-v0"
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "normalize": True,   
        "absolute": True,
    },
    'screen_height': 300,
    'screen_width': 1200,
    'vehicles_count': 50,

}

env = gymnasium.make(envName, config=config, render_mode='human')

# Initialize your model and load parameters
if baseline: 
    agent = BaselineAgent(env)
else: 
    agent = Agent(env)
    checkpoint = torch.load("HighestReward.pth", map_location=torch.device('cpu'))
    #checkpoint = torch.load("ppo_highway_agent.pth", map_location=torch.device('cpu'))
    agent.load_state_dict(checkpoint)
    agent.eval()



# Evaluation loop
state, _ = env.reset()
done, truncated = False, False


fileName = {
    True: 'Baseline',
    False: 'Agent'
}

files = {
    'Data': f'{fileName[baseline]}ControlActions.csv',
    'Rewards': f'{fileName[baseline]}ControlRewards.csv'
}
rewardsHeader = ['Crashed', 'Rewards']
actionsHeader = ['Speed', 'Action']

needsHeader = {key: not os.path.isfile(path) for key, path in files.items()}


#Write on file the inforations
with open(files['Data'], 'a', newline = '') as f1, open(files['Rewards'], 'a', newline = '') as f2:

    dataWriter = csv.writer(f1)
    rewardWriter = csv.writer(f2)

    #Define the headers of the csv files
    if needsHeader['Data']:
        dataWriter.writerow(actionsHeader)
    
    if needsHeader['Rewards']:
        rewardWriter.writerow(rewardsHeader)


    epReward = 0

    while True:

        if baseline: action = agent.BasePolicy(state)
        else: 
            state = torch.Tensor(state).flatten()
            action, _, _, _ = agent.GetActionValue(state)
        
        #Take a step in the simulation
        obs, reward, done, truncated, info = env.step(action)

        dataWriter.writerow([info['speed'], info['action']])

        env.render()

        #Compute final reward
        epReward += reward

        if done or truncated:
            rewardWriter.writerow([info['crashed'], epReward])
            state, _ = env.reset()

env.close()
