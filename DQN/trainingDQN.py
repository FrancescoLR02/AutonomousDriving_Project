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
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

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
    'duration': 80,
    'lanes_count': 3,
    "policy_frequency": 2,
    'right_lane_reward': 0,
    'high_speed_reward': 0.6,
}

env = gymnasium.make(envName, config=config, render_mode=None)


#Hyperparameters
lr = 1e-4
gamma = 0.99
epsStart = 0.9
epsEnd = 0.005
epsDecay = 2500
tau = 0.001

batchSize = 128
numEpisodes = 6000


#initialize the environment
state, info = env.reset()

nActions = env.action_space.n 
stateShape = np.prod(env.observation_space.shape)


policyNet = modelDQN.DQN(stateShape, nActions).to(device)
targetNet = modelDQN.DQN(stateShape, nActions).to(device)

targetNet.load_state_dict(policyNet.state_dict())

optimizer = optim.Adam(policyNet.parameters(), lr = lr, amsgrad=True)
memory = ReplayBuffer.ReplayMemory(capacity=20_000)

steps = 0


episodeRewards = []
losses = []
success = []

#Training:
best_reward = -float('inf')
with open('DQN/DDQNTrainingData1.csv', 'w', newline = '') as f1:
    Data = csv.writer(f1)
    Data.writerow(['Episode', 'Avg Reward', 'Avg Loss', 'SuccessRate', 'Eps'])
    
    for update in range(numEpisodes):

        state, info = env.reset()
        state = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)

        totalReward = 0
        episodeLosses = []

        for t in count():
            action = UtilFunctions.GetAction(env, state, policyNet, device, steps)
            steps += 1

            obs, reward, terminated, truncated, info = env.step(action.item())

            totalReward += reward

            reward = torch.tensor([reward], device = device, dtype = torch.float32)
            done = terminated or truncated

            #If the episode terminated end, otherwise define next state
            if terminated:
                nextState = None
            else:
                nextState = torch.tensor(obs, dtype = torch.float32, device = device).unsqueeze(0)

            #Save the transition
            memory.Push(state, action, nextState, reward)
            
            #Update state
            state = nextState

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

        if update % 100 == 0:
            torch.save(policyNet.state_dict(), "DDQN_policyNet1.pth")
        
        #debug informations
        if update % 100 == 0 and len(episodeRewards) > 0:
            avgRev = np.mean(episodeRewards[-100:])
            avgLoss = np.mean(losses[-100:]) if len(losses) > 0 else 0
            successRate = np.mean(success[-100:]) if len(success) > 0 else 0
            eps = epsEnd + (epsStart - epsEnd) * np.exp(-1 * steps/epsDecay)
            print(f" Episode {update} | Avg Reward: {avgRev:.2f} | Avg Loss: {avgLoss:.4f} | Eps: {eps:.2f} | SuccessRate: {successRate}")

            Data.writerow([update, avgRev, avgLoss, successRate, eps])
            f1.flush()

            if avgRev > best_reward:
                best_reward = avgRev
                torch.save(policyNet.state_dict(), "DDQN_Champion1.pth")
        