import numpy as np
import random
from collections import namedtuple, deque
from itertools import count


#define the map (state, action) pairs to the next (state, reward)
Transition = namedtuple('Transition', ('state', 'action', 'nextState', 'reward'))

class ReplayMemory(object):

   def __init__(self, capacity):
      self.memory = deque([], maxlen = capacity)

   #Save the transition
   def Push(self, *args):
      self.memory.append(Transition(*args))

   #Sample a batch of transition events 
   def Sample(self, batchSize):
      return random.sample(self.memory, batchSize)
   
   def __len__(self):
      return len(self.memory)