import numpy as np
import gymnasium
import highway_env
import torch

from environment import *

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
   'duration': 5,
   "lanes_count": 3,
}



model = Environment()
highwayEnv = model.HighwayEnv(config)
mergerEnv = model.MergerEnv(config)

highwayEnv.reset()

run = True
while run:

   obs, reward, done, truncated, info = highwayEnv.step(highwayEnv.action_space.sample())
   highwayEnv.render()

   if done:
      state, _ = highwayEnv.reset()


   elif truncated:
      finalState = {
         'speed': highwayEnv.unwrapped.vehicle.speed,
      }

      mergerEnv.reset()
      print(finalState['speed'])
      mergerEnv.unwrapped.vehicle.speed = finalState['speed']

      mergerRun = True
      while mergerRun:
         obs, reward, done, truncated, info = mergerEnv.step(mergerEnv.action_space.sample())

         mergerEnv.render()

         if done or truncated:
            state, _ = mergerEnv.reset()
            highwayEnv.reset()
            mergerRun = False

highwayEnv.close()
mergerEnv.close()


