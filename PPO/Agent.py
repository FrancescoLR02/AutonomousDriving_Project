import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class Agent(nn.Module):

   def __init__(self, env):
      super().__init__()

      #Allows for parallelization
      if hasattr(env, "single_observation_space"):
         self.inputDim = np.array(env.single_observation_space.shape).prod()
         self.outputDim = env.single_action_space.n
      else:
         self.inputDim = np.array(env.observation_space.shape).prod()
         self.outputDim = env.action_space.n


      #Actor Network
      self.Actor = nn.Sequential(
         nn.Linear(self.inputDim, 128),
         nn.ReLU(),
         nn.Linear(128, 256),
         nn.ReLU(),
         nn.Linear(256, self.outputDim)
      )

      #Critic Network
      self.Critic = nn.Sequential(
         nn.Linear(self.inputDim, 128),
         nn.ReLU(),
         nn.Linear(128, 256),
         nn.ReLU(),
         nn.Linear(256, 1)
      )

   #Critics prediction
   def GetValue(self, x):
      return self.Critic(x)

   def GetActionValue(self, x, action = None):
      
      logits = self.Actor(x)

      #Output of the network are applied to softmax 
      outputProb = Categorical(logits = logits)

      #If no action is selected, draw it using outputProbs probability dist
      if action is None:
         action = outputProb.sample()

      value = self.Critic(x)

      return action, outputProb.log_prob(action), outputProb.entropy(), value

