import numpy as np
import gymnasium
import highway_env
from itertools import count
import random
import tqdm as tqdm



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import modelDQN
import ReplayBuffer
import UtilFunctions

np.set_printoptions(linewidth=300, suppress=True, precision=5)

device = torch.device('cpu')

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
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": True,   
        "absolute": False,
    },
    'duration': 80,
    'lanes_count': 3,
    "policy_frequency": 1,
}

env = gymnasium.make(envName, config=config, render_mode=None)


#Hyperparameters
lr = 3e-4
gamma = 0.99
epsStart = 0.9
epsEnd = 0.01
epsDecay = 2500
tau = 0.005

batchSize = 64
numEpisodes = 100


#initialize the environment
state, info = env.reset()

nActions = env.action_space.n 
stateShape = np.prod(env.observation_space.shape)


policyNet = modelDQN.DQN(stateShape, nActions).to(device)
targetNet = modelDQN.DQN(stateShape, nActions).to(device)

targetNet.load_state_dict(policyNet.state_dict())

optimizer = optim.Adam(policyNet.parameters(), lr = lr)
memory = ReplayBuffer.ReplayMemory(capacity=10_000)

steps = 0


#Training:

for update in tqdm.tqdm(range(numEpisodes)):

    state, info = env.reset()
    state = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)

    for t in count():
        action = UtilFunctions.GetAction(env, state, policyNet, device, steps)
        steps += 1

        obs, reward, terminated, truncated, _ = env.step(action.item())

        reward = torch.tensor([reward], device = device)
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
        UtilFunctions.Optimizer(memory, policyNet, targetNet, optimizer, device)

        #Update network weights:
        targetNet_stateDict = targetNet.state_dict()
        policyNet_stateDict = policyNet.state_dict()

        for k in policyNet_stateDict:
            targetNet_stateDict[k] = policyNet_stateDict[k]*tau + targetNet_stateDict[k]*(1-tau)
        
        targetNet.load_state_dict(targetNet_stateDict)

        if done:
            break

    if update % 10 == 0:
        torch.save(policyNet.state_dict(), "../DQN_policyNet.pth")


        