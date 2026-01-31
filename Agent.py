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


      self.Network = nn.Sequential(
         nn.Linear(self.inputDim, 32), 
         nn.ReLU(),
         nn.Linear(32, 64),
         nn.ReLU()
      )

      self.Actor = nn.Linear(64, self.outputDim)
      self.Critic = nn.Linear(64, 1)

   #Critics prediction
   def GetValue(self, x):
      #x = x.flatten(start_dim = 1)
      self.Critic(self.Network(x))

   def GetActionValue(self, x, action = None):
      
      #x = x.flatten(start_dim = 1)
      hiddenLayers = self.Network(x)

      #Output of the network are applied to softmax 
      outputs = self.Actor(hiddenLayers)
      outputProb = Categorical(logits = outputs)

      #If no action is selected, draw it using outputProbs probability dist
      if action is None:
         action = outputProb.sample()

      value = self.Critic(hiddenLayers)

      return action, outputProb.log_prob(action), outputProb.entropy(), value

