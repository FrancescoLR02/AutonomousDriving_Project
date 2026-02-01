import gymnasium
import highway_env
import numpy as np
import torch
import random
from tqdm import tqdm

#from Agent import *

np.set_printoptions(linewidth=200, suppress=True, precision=5)

device = torch.device('cpu')


# Set the seed and create the environment
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


def main():
   envName = "highway-v0"
   config = {
      "observation": {
         "type": "Kinematics",
         "vehicles_count": 10,
         "features": ["presence", "x", "y", "vx", "vy", 'cos_h', 'sin_h'],
         "normalize": True,   
         "absolute": True,
      },

      "policy_frequency": 5,
      'vehicles_count': 50, 
      'vehicles_density': 1.2,
      'collision_reward': -2.0,
      'lane_change_reward': 0.4,

   }

   numEvents = 2
   envs = gymnasium.make_vec(envName, config = config, num_envs=numEvents, vectorization_mode="async")



   state, info = envs.reset()
   nextState, reward, terminated, truncated, _ = envs.step(envs.action_space.sample())

   print(state)
   print(nextState)



if __name__ == '__main__':

   main()