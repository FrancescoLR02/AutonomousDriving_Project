import numpy as np
import gymnasium
import highway_env
import torch 



HighwayConfig = {
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
      'duration': 20,
      "lanes_count": 3,
   }

MergerConfig = {
         "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "normalize": True,   
            "absolute": False,
         },
         "action":{
            "type": "DiscreteMetaAction",
            "target_speeds": [12, 15, 18, 21, 24], 
         },
         'screen_height': 300,
         'screen_width': 1200,
         'duration': 20,
         "lanes_count": 3,
      }



class Environment:

   def __init__(self, changeEnv = 200):
      self.changeEnv = changeEnv

   
   def HighwayEnv(self, renderMode = None):
      
      env = gymnasium.make('highway-v0', config = HighwayConfig,  render_mode=renderMode)
      return env

   def MergerEnv(self, renderMode = None):
      env = gymnasium.make('merge-v0', config = MergerConfig, render_mode=renderMode)
      return env
