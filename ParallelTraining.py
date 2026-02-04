import gymnasium
import highway_env
import numpy as np
import torch
import random
from tqdm import tqdm

from Agent import *

np.set_printoptions(linewidth=200, suppress=True, precision=5)

device = torch.device('cpu')
#device = torch.backends.mps.is_available('mps')


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
         "absolute": False,
      },

      "policy_frequency": 2,
      'vehicles_count': 50, 
      'vehicles_density': 1.1,
      'collision_reward': -5.0,
      'high_speed_reward': 0.5,
      'right_lane_reward': 0.1,
      'lane_change_reward': 0.1,
      'duration': 60,
      'reward_speed_range': [20, 30]


   }

   numEvents = 5
   envs = gymnasium.make_vec(envName, config = config, num_envs=numEvents, vectorization_mode="async")

   lr = 1e-4
   gamma = 0.99
   gaeLambda = 0.95
   clipCoeff = 0.2

   MAX_STEPS = int(5e5) 
   numSteps = 500 # 256*numEvents
   batchSize = 100

   agent = Agent(envs).to(device)
   optimizer = optim.Adam(agent.parameters(), lr = lr)

   stateShape = np.prod(envs.observation_space.shape[1:])

   stateBuffer = torch.zeros((numSteps, numEvents, stateShape)).to(device)
   actionBuffer = torch.zeros((numSteps, numEvents)).to(device)
   logProbBuffer = torch.zeros((numSteps, numEvents)).to(device)
   rewardBuffer = torch.zeros((numSteps, numEvents)).to(device)
   doneBuffer = torch.zeros((numSteps, numEvents)).to(device)
   valuesBuffer = torch.zeros((numSteps, numEvents)).to(device)


   nextState, info = envs.reset()
   nextState = torch.Tensor(nextState).flatten().reshape((numEvents, stateShape)).to(device)
   nextDone = torch.zeros(numEvents).to(device)

   MAX_UPDATES = MAX_STEPS // (numSteps * numEvents)

   for step in range(MAX_UPDATES):

      for i in range(numSteps):

         stateBuffer[i] = nextState
         doneBuffer[i] = nextDone

         with torch.no_grad():
            actions, logProbs, _, values = agent.GetActionValue(nextState)

            valuesBuffer[i] = values.flatten()
            actionBuffer[i] = actions
            logProbBuffer[i] = logProbs


         nextState, reward, terminated, truncated, _ = envs.step(actions.cpu().numpy())

         done = torch.tensor(terminated | truncated).to(device).float()
         nextState = torch.Tensor(nextState).flatten().reshape((numEvents, stateShape)).to(device)
         nextDone = done


         reward = torch.tensor(reward).to(device).view(-1)
         truncated = torch.tensor(truncated).to(device).float()

         # velocityReward = truncated*stateBuffer[i][:, 3]

         # #Overtake logic
         # currentState3D = stateBuffer[i].view(numEvents, 10, 7)
         # neighborVx = currentState3D[:, 1:, 3]
         # passingFlow = torch.mean(torch.clamp(-neighborVx, min=0), dim=1)
         # overtakeReward = passingFlow

         #Bonus if car goes fast
         rewardBuffer[i] = reward# + overtakeReward*5 #+ velocityReward*2 + 


      #Compute the next value in the next state
      with torch.no_grad():
         _, _, _, nextValue = agent.GetActionValue(nextState)
         nextValue = nextValue.reshape(-1)

      advantage = torch.zeros_like(rewardBuffer).to(device)
      lastGAElam = torch.zeros(numEvents).to(device)

      for t in reversed(range(numSteps)):
         if t == numSteps - 1:
            # we calculated outside the loop to bootstrap.
            nextNonTerminal = 1.0 - nextDone
            nextValues = nextValue.reshape(-1)
         # Otherwise, we look at the next tep in the buffer
         else:
            nextNonTerminal = 1.0 - doneBuffer[t + 1]
            nextValues = valuesBuffer[t + 1]

         #element-wise operation on tensors of shape
         delta = rewardBuffer[t] + gamma * nextValues * nextNonTerminal - valuesBuffer[t]
         advantage[t] = lastGAElam = delta + gamma * gaeLambda * nextNonTerminal * lastGAElam

      #Calculate Returns
      returns = advantage + valuesBuffer
      advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

      #Training on the batch of observations
      batchState = stateBuffer.reshape((-1, stateShape))
      batchLogProb = logProbBuffer.reshape(-1)
      batchActions = actionBuffer.reshape(-1)
      batchAdvantages = advantage.reshape(-1)
      batchReturns = returns.reshape(-1)
      batchValues = valuesBuffer.reshape(-1)

      totSamples = batchState.size(0)
      idxs = np.arange(totSamples)

      for epoch in range(3):

         np.random.shuffle(idxs)

         for start in range(0, totSamples, batchSize):
            
            #Select the batch indexes data
            end = start + batchSize
            idx = idxs[start : end]

            #Evaluate the batch of paris state-action
            _, newLogProb, entropy, newValue = agent.GetActionValue(batchState[idx], batchActions[idx])

            #Compute log ratio:
            logRatio = newLogProb - batchLogProb[idx]
            ratio = logRatio.exp()

            #Policy loss normalized
            mbAdvantage = batchAdvantages[idx]

            policyLoss1 = -mbAdvantage * ratio
            policyLoss2 = -mbAdvantage * torch.clamp(ratio, 1 - clipCoeff, 1 + clipCoeff)
            
            #Expectation value of PPO objective
            policyLoss = torch.max(policyLoss1, policyLoss2).mean()

            #value Loss 
            vLoss = 1/2 * ((newValue.view(-1) - batchReturns[idx]) ** 2).mean()

            #Total loss
            loss = policyLoss - 0.02*entropy.mean() + 0.5*vLoss

            #Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
            optimizer.step()

      MeanReward = rewardBuffer.sum()

      print(f"Update {step+1}/{MAX_UPDATES} | Loss: {loss.item():.4f} | Mean Reward: {MeanReward:.2f}")

      torch.save(agent.state_dict(), "ppo_highway_agent1.pth")



if __name__ == '__main__':

   main()