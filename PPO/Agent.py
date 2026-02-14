import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


def LayerInit(layer, std=np.sqrt(2), bias_const=0.0):
   torch.nn.init.orthogonal_(layer.weight, std)
   torch.nn.init.constant_(layer.bias, bias_const)
   return layer


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
         LayerInit(nn.Linear(self.inputDim, 256), std = np.sqrt(2)),
         nn.ReLU(),
         LayerInit(nn.Linear(256, 128), std = np.sqrt(2)),
         nn.ReLU()
      )

      self.Actor = LayerInit(nn.Linear(128, self.outputDim), std=0.1)
      self.Critic = LayerInit(nn.Linear(128, 1), std = 1)

   #Critics prediction
   def GetValue(self, x):
      #x = x.flatten(start_dim = 1)
      return self.Critic(self.Network(x))

   def GetActionValue(self, x, action = None, actionMask = None):
      
      #x = x.flatten(start_dim = 1)
      hiddenLayers = self.Network(x)
      logits = self.Actor(hiddenLayers)

      #Mask not available action
      if actionMask is not None:
         actionMask = torch.as_tensor(actionMask, dtype = torch.bool, device = logits.device)
         
         #Assign low value to illegal actions
         logits = logits.masked_fill(~actionMask, -1e8)

      #Output of the network are applied to softmax 
      outputProb = Categorical(logits = logits)

      #If no action is selected, draw it using outputProbs probability dist
      if action is None:
         action = outputProb.sample()

      value = self.Critic(hiddenLayers)

      return action, outputProb.log_prob(action), outputProb.entropy(), value

