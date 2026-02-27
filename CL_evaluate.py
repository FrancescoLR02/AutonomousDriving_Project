import numpy as np
import gymnasium
import highway_env
import torch
import os
import csv

from Continual_Learning.environment import *
import DQN.modelDQN



#pid = os.getpid()

rm = 'human' 

model = Environment()
highwayEnv = model.HighwayEnv(renderMode=rm)
mergerEnv = model.MergerEnv(renderMode=rm)

state, _ = highwayEnv.reset()

nActions = highwayEnv.action_space.n 
stateShape = np.prod(highwayEnv.observation_space.shape)

currModel = torch.load(f'CL_Champion_merge-v0.pth')
agent = DQN.modelDQN.DQN(stateShape, nActions)

agent.load_state_dict(currModel)
agent.eval()


#Save on informations on file 
fileName = 'CL_DQN_agent'

files = {
   'Rewards': f'Data/{fileName}ControlRewards.csv'
}
rewardsHeader = ['HighwayCrashed', 'HighwayRewards', 'MergerCrashed', 'MergerRewards']

needsHeader = {key: not os.path.isfile(path) for key, path in files.items()}



with open(files['Rewards'], 'a', newline = '') as f1:

   rewardWriter = csv.writer(f1)
   
   if needsHeader['Rewards']:
      rewardWriter.writerow(rewardsHeader)



   run = True
   while run:
      #Define variable for the event
      crashedHighway = False
      crashedMerger = False
      rewardHighway = 0
      rewardMerger = 0

      #Reset highway to start the new run
      state, _ = highwayEnv.reset()

      #Highway
      highwayRun = True
      while highwayRun:
         state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

         with torch.no_grad():
            qValue = agent(state_tensor)
            action = qValue.max(1).indices.item()

         nextState, reward, done, truncated, info = highwayEnv.step(action)

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

            with torch.no_grad():
               qValue = agent(mState_tensor)
               action = qValue.max(1).indices.item()

            obs, reward, done, truncated, info = mergerEnv.step(action)

            print(info['speed'])

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

      rewardWriter.writerow([crashedHighway, rewardHighway, crashedMerger, rewardMerger])

   highwayEnv.close()
   mergerEnv.close()