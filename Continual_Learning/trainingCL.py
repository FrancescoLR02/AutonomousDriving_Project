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

import DQN.modelDQN
import DQN.ReplayBuffer
import DQN.UtilFunctions

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
   'right_lane_reward': 0,
   'high_speed_reward': 0.6,
}


tasksCL = ['highway-v0', 'merge-v0']
epsPerTask = 4_000

lr = 1e-4
tau = 0.001

memory = DQN.ReplayBuffer.ReplayMemory(capacity=250_000)

#Actions: 5, stateSpace = 7*10
nActions = 5
stateShape = 70

policyNet = DQN.modelDQN.DQN(stateShape, nActions).to(device)
targetNet = DQN.modelDQN.DQN(stateShape, nActions).to(device)

optimizer = optim.Adam(policyNet.parameters(), lr = lr, amsgrad=True)
targetNet.load_state_dict(policyNet.state_dict())


#Train both environment
for taskID, envName in enumerate(tasksCL):

   print(f'Training on {envName} environment')

   env = gymnasium.make(envName, config=config, render_mode=None)

   steps = 0

   episodeRewards = []
   success = []

   #Training on the single environment:
   best_reward = -float('inf')
   for update in range(epsPerTask):

      state, info = env.reset()
      state = state.flatten()

      totalReward = 0

      for t in count():

         stateTensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
         action = DQN.UtilFunctions.GetAction(env, stateTensor, policyNet, device, steps)
         steps += 1

         obs, reward, terminated, truncated, info = env.step(action.item())

         totalReward += reward

         reward = torch.tensor([reward], device = device, dtype = torch.float32)
         done = terminated or truncated

         #If the episode terminated end, otherwise define next state
         if terminated:
            nextState = None
         else:
            nextState = obs.flatten()

         #Save the transition
         memory.Push(state, action.item(), nextState, float(reward))
         
         #Update state
         state = nextState

         #Perform one step in optimization
         lossValue = DQN.UtilFunctions.Optimizer(memory, policyNet, targetNet, optimizer, device)

         #Update network weights:
         targetNet_stateDict = targetNet.state_dict()
         policyNet_stateDict = policyNet.state_dict()

         for k in policyNet_stateDict:
            targetNet_stateDict[k] = policyNet_stateDict[k]*tau + targetNet_stateDict[k]*(1-tau)
         
         targetNet.load_state_dict(targetNet_stateDict)

         if done:
            success.append(not(info['crashed']))
            episodeRewards.append(totalReward)
            break

      if update % 50 == 0:
            torch.save(policyNet.state_dict(), f"Continual_Learning/Models/CLpolicyNet_{update}_{envName}.pth")

      if update == 0:
            print(f"{'Update':<8} | {'AvgReward':<5} | {'SuccessRate':<5} |")
            print("-" * 75)
      
      #debug informations
      if update % 50 == 0 and len(episodeRewards) > 0:
         avgRev = np.mean(episodeRewards[-50:])
         successRate = np.mean(success[-50:]) if len(success) > 0 else 0
         print(f"{update:<8} | {avgRev:.5f} | {successRate:.5f}")


         if avgRev > best_reward:
            best_reward = avgRev
            torch.save(policyNet.state_dict(), f"CL_Champion_{envName}.pth")
