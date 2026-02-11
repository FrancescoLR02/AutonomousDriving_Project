import numpy as np
import gymnasium
import highway_env
import torch 



class Environment:

   def __init__(self, changeEnv = 200):
      self.changeEnv = changeEnv

   
   def HighwayEnv(self, config):
      
      env = gymnasium.make('highway-v0', config = config,  render_mode='rgb_array')
      return env

   def MergerEnv(self, config):
      env = gymnasium.make('merge-v0', config = config, render_mode='rgb_array')
      return env
