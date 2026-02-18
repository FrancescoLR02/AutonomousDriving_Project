import numpy as np
import gymnasium
import highway_env
import torch
import random
import os 
import csv

from baseline import BaselineAgent
from PPO.Agent import *

np.set_printoptions(linewidth=200, suppress=True, precision=5)


# Set the seed and create the environment
np.random.seed(0)
random.seed(0)
#torch.Agent_seed(0)

baseline = True

pid = os.getpid()


envName = "highway-v0"
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", 'cos_h', 'sin_h'],
        "normalize": True,   
        "absolute": False,
    },
    'screen_height': 300,
    'screen_width': 1200,
    "policy_frequency": 2,
    'duration': 80,
}

env = gymnasium.make(envName, config=config, render_mode=None)

# Initialize your model and load parameters
if baseline: 
    agent = BaselineAgent(env)
else: 
    agent = Agent(env)
    checkpoint = torch.load("singleTraining.pth", map_location=torch.device('cpu'))
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
    'Data': f'Data/{fileName[baseline]}ControlActions_{pid}.csv',
    'Rewards': f'Data/{fileName[baseline]}ControlRewards_{pid}.csv'
}
rewardsHeader = ['Crashed', 'Rewards', 'AvgSpeed', 'StdSpeed']
actionsHeader = ['Speed', 'Action']

needsHeader = {key: not os.path.isfile(path) for key, path in files.items()}


#Write on file the inforations
with  open(files['Rewards'], 'a', newline = '') as f2: #open(files['Data'], 'a', newline = '') as f1,

    #dataWriter = csv.writer(f1)
    rewardWriter = csv.writer(f2)

    #Define the headers of the csv files
    # if needsHeader['Data']:
    #     dataWriter.writerow(actionsHeader)
    
    if needsHeader['Rewards']:
        rewardWriter.writerow(rewardsHeader)


    epReward = 0
    episode = 0

    avgSpeed = []

    while True:
        episode += 1

        if baseline: action = agent.BasePolicy(state)
        else: 
            state = torch.as_tensor(state, dtype=torch.float32).flatten().unsqueeze(0)

            with torch.no_grad():

                hidden = agent.Network(state)
                logits = agent.Actor(hidden)
                action = torch.argmax(logits).item()
                    
        #Take a step in the simulation
        nextState, reward, done, truncated, info = env.step(action)

        #dataWriter.writerow([info['speed'], info['action']])
        avgSpeed.append(info['speed'])

        #env.render()

        #Compute final reward
        epReward += reward

        #update state
        state = nextState

        if done or truncated:
            rewardWriter.writerow([info['crashed'], epReward, np.mean(avgSpeed), np.std(avgSpeed)])
            state, _ = env.reset()
            avgSpeed = []
            epReward = 0
            #f1.flush()
            f2.flush()


env.close()
