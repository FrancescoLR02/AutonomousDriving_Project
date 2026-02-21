import numpy as np
import gymnasium
import highway_env
from itertools import count
import random
import tqdm as tqdm
import csv



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import modelDQN
import ReplayBuffer
import UtilFunctions

np.set_printoptions(linewidth=300, suppress=True, precision=5)

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

print(device)

# Set the seed and create the environment
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

#Config and environment
envName = "highway-v0"
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", 'cos_h', 'sin_h'],
        "normalize": True,   
        "absolute": False,
    },

    "action":{
        "type": "DiscreteMetaAction",
        "target_speeds": [18, 21, 24, 27, 30], 
    },
    'duration': 40,
    'lanes_count': 3,
    "policy_frequency": 2,
    'high_speed_reward': 0.4,
    'right_lane_reward':0
}

env = gymnasium.make(envName, config=config, render_mode=None)


#Hyperparameters
lr = 1e-4
gamma = 0.9
epsStart = 0.9
epsEnd = 0.01
epsDecay = 30_000
tau = 0.005

batchSize = 64
numEpisodes = 6000


#initialize the environment
state, info = env.reset()

nActions = env.action_space.n 
stateShape = np.prod(env.observation_space.shape)


policyNet = modelDQN.DQN(stateShape, nActions).to(device)
targetNet = modelDQN.DQN(stateShape, nActions).to(device)

targetNet.load_state_dict(policyNet.state_dict())

optimizer = optim.Adam(policyNet.parameters(), lr = lr, amsgrad=True)
memory = ReplayBuffer.ReplayMemory(capacity=100_000)

steps = 0


episodeRewards = []
losses = []
success = []
speed = []

#Training:
best_reward = -float('inf')
with open('DQN/DDQNTrainingData1.csv', 'w', newline = '') as f1:
    Data = csv.writer(f1)
    Data.writerow(['Episode', 'Avg Reward', 'Avg Loss', 'SuccessRate', 'Eps'])
    
    for update in range(numEpisodes):

        state, info = env.reset()
        #state = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
        state = state.flatten()

        totalReward = 0
        episodeLosses = []

        for t in count():
            stateTensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = UtilFunctions.GetAction(env, stateTensor, policyNet, device, steps, epsDecay=epsDecay)
            steps += 1

            obs, reward, terminated, truncated, info = env.step(action.item())

            totalReward += reward
            speed.append(info['speed'])

            reward = torch.tensor([reward], device = device, dtype = torch.float32)
            done = terminated or truncated

            #If the episode terminated end, otherwise define next state
            if terminated:
                nextState = None
            else:
                #nextState = torch.tensor(obs, dtype = torch.float32, device = device).unsqueeze(0)
                nextState = obs.flatten()

            #Save the transition
            memory.Push(state, action.item(), nextState, float(reward))
            
            #Update state
            state = nextState

            #Optimize every 4 steps
            if steps % 1 == 0 and len(memory) > 1000:

                #Perform one step in optimization
                lossValue = UtilFunctions.Optimizer(memory, policyNet, targetNet, optimizer, device)

                if lossValue is not None:
                    episodeLosses.append(lossValue)

                #Update network weights:
                targetNet_stateDict = targetNet.state_dict()
                policyNet_stateDict = policyNet.state_dict()

                for k in policyNet_stateDict:
                    targetNet_stateDict[k] = policyNet_stateDict[k]*tau + targetNet_stateDict[k]*(1-tau)
                
                targetNet.load_state_dict(targetNet_stateDict)

            if done:
                success.append(not(info['crashed']))
                episodeRewards.append(totalReward)
                if len(episodeLosses) > 0:
                    losses.append(np.mean(episodeLosses))
                break

        if update % 50 == 0:
            torch.save(policyNet.state_dict(), f"DQN/Models/DDQN_policyNet_{update}.pth")

        if update == 0:
            print(f"{'Update':<8} | {'AvgReward':<5} | {'Avg Spd':<5} | {'Avg loss':<5} | {'Eps':<5} | {'SuccessRate':<5} |")
            print("-" * 75)
        
        #debug informations
        if update % 50 == 0 and len(episodeRewards) > 0:
            avgRev = np.mean(episodeRewards[-50:])
            avgLoss = np.mean(losses[-50:]) if len(losses) > 0 else 0
            avgSpeed = np.mean(speed[-50:]) if len(speed) > 0 else 0
            successRate = np.mean(success[-50:]) if len(success) > 0 else 0
            eps = epsEnd + (epsStart - epsEnd) * np.exp(-1 * steps/epsDecay)
            print(f"{update:<8} | {avgRev:.5f} | {avgSpeed:.5f} | {avgLoss:.5f} | {eps:.5f} | {successRate:.5f}")

            Data.writerow([update, avgRev, avgLoss, successRate, eps])
            f1.flush()

            if avgRev > best_reward:
                best_reward = avgRev
                torch.save(policyNet.state_dict(), "DDQN_Champion1.pth")
        