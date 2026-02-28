import numpy as np
import gymnasium
import highway_env
import torch
import random
import os 
import csv
import sys

from baseline import BaselineAgent
from PPO.Agent import *

np.set_printoptions(linewidth=200, suppress=True, precision=5)


# Set the seed and create the environment
np.random.seed(0)
random.seed(0)
#torch.Agent_seed(0)

if len(sys.argv) == 1:
    baseline = False

else:
    baseline = bool(sys.argv[1])

pid = os.getpid()


envName = "highway-v0"
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "normalize": True,   
        "absolute": False,
    },

    "action":{
        "type": "DiscreteMetaAction",
        "target_speeds": [18, 21, 24, 27, 30], 
    },
    'screen_height': 300,
    'screen_width': 1200,
    "policy_frequency": 2,
    'duration': 40,
    'vehicle_density': 0.8
}

env = gymnasium.make(envName, config=config, render_mode='human')

# Initialize your model and load parameters
if baseline: 
    agent = BaselineAgent(env)
else: 
    agent = Agent(env)
    checkpoint = torch.load("PPO_Champion.pth", map_location=torch.device('cpu'))
    agent.load_state_dict(checkpoint)
    agent.eval()



# Evaluation loop
state, _ = env.reset()
done, truncated = False, False


epReward = 0
episode = 0

avgSpeed = []

while True:
    episode += 1

    if baseline: action = agent.BasePolicy(state)
    else: 
        state = torch.as_tensor(state, dtype=torch.float32).flatten().unsqueeze(0)

        with torch.no_grad():

            logits = agent.Actor(state)
            action = torch.argmax(logits).item()
                
    #Take a step in the simulation
    nextState, reward, done, truncated, info = env.step(action)

    avgSpeed.append(info['speed'])
    env.render()
    #Compute final reward
    epReward += reward

    #update state
    state = nextState

    if done or truncated:
        #rewardWriter.writerow([info['crashed'], epReward, np.mean(avgSpeed), np.std(avgSpeed)])
        state, _ = env.reset()
        avgSpeed = []
        epReward = 0
        #f2.flush()


env.close()
