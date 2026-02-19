import numpy as np
import torch
import torch.nn as nn

from ReplayBuffer import *


epsStart = 0.9
epsEnd = 0.01
#epsDecay = 2500
gamma = 0.99
batchSize = 64

def GetAction(env, state, policyNet, device, update, epsDecay = 2500):

   sample = np.random.random()
   epsTH = epsEnd + (epsStart - epsEnd) * np.exp(-1 * update/epsDecay)

   if sample > epsTH:
      with torch.no_grad():
         return policyNet(state).max(1).indices.view(1, 1)
      
   else:
      return torch.tensor([[env.action_space.sample()]], device = device)
   


def Optimizer(memory, policyNet, targetNet, optimizer, device):

   if len(memory) < batchSize:
      return
   
   transitions = memory.Sample(batchSize)

   #Transposition of the batch tensor
   batch = Transition(*zip(*transitions))

   #Mask of non-final states 
   nonFinal_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.nextState)), device = device, dtype = torch.bool)

   #Apply the mask to the states:
   validNextState = [s for s in batch.nextState if s is not None]
   nonFinal_nextStates = torch.tensor(np.array(validNextState), dtype = torch.float32, device = device)

   #Define the batches 
   stateBatch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
   actionBatch = torch.tensor(batch.action, dtype=torch.int64, device=device).unsqueeze(1)
   rewardBatch = torch.tensor(batch.reward, dtype=torch.float32, device=device)

   #Compute the action-value function
   stateActionValue = policyNet(stateBatch).gather(1, actionBatch)

   #Compute the value function at next state
   stateValueNext = torch.zeros(batchSize, device = device)
   with torch.no_grad():

      #DDQN 
      bestNextActions = policyNet(nonFinal_nextStates).max(1).indices.unsqueeze(1)
      stateValueNext[nonFinal_mask] = targetNet(nonFinal_nextStates).gather(1, bestNextActions).squeeze(1)

   #Expected Q value
   expStateActionValue = (stateValueNext * gamma) + rewardBatch

   #Huber loss:
   criterion = nn.SmoothL1Loss()
   loss = criterion(stateActionValue, expStateActionValue.unsqueeze(1))

   #Optimizatino
   optimizer.zero_grad()
   loss.backward()
   torch.nn.utils.clip_grad_value_(policyNet.parameters(), 100)
   optimizer.step()

   return loss.item()