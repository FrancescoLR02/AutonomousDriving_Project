import numpy as np
import gymnasium

import torch
import torch.nn as nn
import torch.nn.functional as F



class DQN(nn.Module):

   def __init__(self, stateShape, nActions):
      super().__init__()

      self.layer1 = nn.Linear(stateShape, 128)
      self.layer2 = nn.Linear(128, 128)
      self.layer3 = nn.Linear(128, nActions)

   def forward(self, x):
      x = x.view(x.size(0), -1)
      x = F.relu(self.layer1(x))
      x = F.relu(self.layer2(x))

      return self.layer3(x)