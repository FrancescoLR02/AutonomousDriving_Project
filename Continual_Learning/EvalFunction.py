import numpy as np
import gymnasium
import highway_env
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from environment import *
import DQN.modelDQN


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
      "target_speeds": [18, 21, 24, 27], 
    },
   'screen_height': 300,
   'screen_width': 1200,
   'duration': 30,
   "lanes_count": 3,
}

def Evaluate(update, envName, config, nEval = 10):

   model = Environment()
   highwayEnv = model.HighwayEnv(config)
   mergerEnv = model.MergerEnv(config)

   # Same for both highway and merger
   nActions = highwayEnv.action_space.n 
   stateShape = np.prod(highwayEnv.observation_space.shape)
   
   currModel = torch.load(f'Continual_Learning/Models/CLpolicyNet_{update}_{envName}.pth')
   agent = DQN.modelDQN.DQN(stateShape, nActions)

   agent.load_state_dict(currModel)
   agent.eval()

   resDict = {
      'Crashed': ([], []),
      'Rewards': ([], [])
   }

   run = 0

   while run < nEval:
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
      resDict['Crashed'][0].append(crashedHighway)
      resDict['Crashed'][1].append(crashedMerger)

      resDict['Rewards'][0].append(rewardHighway)
      resDict['Rewards'][1].append(rewardMerger)

      run += 1

   highwayEnv.close()
   mergerEnv.close()
   
   return resDict