import numpy as np
import gymnasium
import highway_env
import torch
import os
import csv
import sys

from Continual_Learning.environment import *
import DQN.modelDQN
from baseline import *



pid = os.getpid()

if len(sys.argv) == 1:
   Baseline = False

else:
   Baseline = bool(sys.argv[1])

rm = 'human'

model = Environment()
highwayEnv = model.HighwayEnv(renderMode=rm)
mergerEnv = model.MergerEnv(renderMode=rm)

state, _ = highwayEnv.reset()

if Baseline: 
   Hagent = BaselineAgent(highwayEnv)
   Magent = BaselineAgent(mergerEnv)

else:
   nActions = highwayEnv.action_space.n 
   stateShape = np.prod(highwayEnv.observation_space.shape)

   currModel = torch.load(f'CL_Champion_merge-v0.pth')
   agent = DQN.modelDQN.DQN(stateShape, nActions)

   agent.load_state_dict(currModel)
   agent.eval()






run = True
while run:
   #Define variable for the event
   crashedHighway = False
   crashedMerger = False
   rewardHighway = 0
   rewardMerger = 0

   Hspeed, Mspeed = [], []

   #Reset highway to start the new run
   state, _ = highwayEnv.reset()

   #Highway
   highwayRun = True
   while highwayRun:
      state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

      if Baseline: action = Hagent.BasePolicy(state)
      
      else:
         with torch.no_grad():
            qValue = agent(state_tensor)
            action = qValue.max(1).indices.item()

      nextState, reward, done, truncated, info = highwayEnv.step(action)

      Hspeed.append(info['speed'])

      highwayEnv.render()

      rewardHighway += reward
      state = nextState

      #If crash or time runs out, end the highway inner loop
      if done or truncated:
         crashedHighway = info.get('crashed', False)
         highwayRun = False


   #Merger
   # Only run the merger if the highway was successfully completed
   if truncated and not crashedHighway:
      finalState = {'speed': highwayEnv.unwrapped.vehicle.speed}
      
      mState, _ = mergerEnv.reset()
      mergerEnv.unwrapped.vehicle.speed = finalState['speed']

      mergerRun = True
      while mergerRun:
         mState_tensor = torch.as_tensor(mState, dtype=torch.float32).unsqueeze(0)

         if Baseline: action = Magent.BasePolicy(mState)

         else:
            with torch.no_grad():
               qValue = agent(mState_tensor)
               action = qValue.max(1).indices.item()

         obs, reward, done, truncated, info = mergerEnv.step(action)

         Mspeed.append(info['speed'])

         mergerEnv.render()
         rewardMerger += reward
         mState = obs

         if done or truncated:
            crashedMerger = info.get('crashed', False)
            mergerRun = False
   else:
      #If crashed on the highway, faills the merger too
      crashedMerger = True
      rewardMerger = 0.0

   #Save results

   if len(Hspeed) == 0: HSpeedAvg = 0
   else: HSpeedAvg = np.mean(Hspeed) 

   if len(Mspeed) == 0: MSpeedAvg = 0
   else: MSpeedAvg = np.mean(Mspeed) 

highwayEnv.close()
mergerEnv.close()