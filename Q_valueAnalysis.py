import numpy as np
import gymnasium
import highway_env
import torch
import random
import os 
import csv
import sys

from DQN.modelDQN import DQN



np.set_printoptions(linewidth=200, suppress=True, precision=5)


# Set the seed and create the environment
np.random.seed(0)
random.seed(0)

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
   'duration': 40,
   "policy_frequency": 2
}

env = gymnasium.make(envName, config=config, render_mode='human')

state, _ = env.reset()

nActions = env.action_space.n 
stateShape = np.prod(env.observation_space.shape)

agent = DQN(stateShape, nActions)
checkpoint = torch.load("DDQN_Champion.pth", map_location=torch.device('cpu'))
agent.load_state_dict(checkpoint)
agent.eval()



done, truncated = False, False

#Define the name of the file
fileName = 'Q_ValueDQN'
files = {
   'Data': f'Data/{fileName}.csv',
}

#Define the header information
Q_ValHeader = ['SimNumber', 'InstantReward', 'InstantSpeed', 'LaneLeft', 'Idle', 'LaneRight', 'Faster', 'Slower', 'Action']

needsHeader = {key: not os.path.isfile(path) for key, path in files.items()}


#Write on file the inforations
with open(files['Data'], 'a', newline = '') as f2:

   Data = csv.writer(f2)
   
   #Define if it needs an header
   if needsHeader['Data']:
      Data.writerow(Q_ValHeader)


   SimNumber = 0

   while True:
      SimNumber += 1

      state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

      with torch.no_grad():

         qValue = agent(state)
         action = qValue.max(1).indices.item()

         qList = qValue.squeeze().tolist()
                  
      #Take a step in the simulation
      nextState, reward, done, truncated, info = env.step(action)


      Data.writerow([SimNumber, reward, info['speed'], qList[0], qList[1], qList[2], qList[3], qList[4], action])

      env.render()

      #update state
      state = nextState

      if done or truncated:
         state, _ = env.reset()
         f2.flush()


env.close()
