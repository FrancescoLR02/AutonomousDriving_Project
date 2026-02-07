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
         "features": ["presence", "x", "y", "vx", "vy"],
         "normalize": True,   
         "absolute": False,
      },
      'duration': 60,
      "policy_frequency": 5,
      'collision_reward': -1,
      'high_speed_reward': 1,
      'right_lane_reward': 0,
      'lane_change_reward': 1,
      'reward_speed_range': [20, 30]


   }

   numEvents = 5
   envs = gymnasium.make_vec(envName, config = config, num_envs=numEvents, vectorization_mode="async")

   lr = 3e-4
   gamma = 0.99
   gaeLambda = 0.95
   clipCoeff = 0.2

   MAX_STEPS = int(1e6) 
   numSteps = 1024
   batchSize = 512

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
         #doneBuffer[i] = nextDone

         with torch.no_grad():
            actions, logProbs, _, values = agent.GetActionValue(nextState)

            valuesBuffer[i] = values.flatten()
            actionBuffer[i] = actions
            logProbBuffer[i] = logProbs


         nextState, reward, terminated, truncated, _ = envs.step(actions.cpu().numpy())

         done = torch.tensor(terminated | truncated).to(device).float()
         doneBuffer[i] = done
         nextState = torch.Tensor(nextState).flatten().reshape((numEvents, stateShape)).to(device)

         reward = torch.tensor(reward).to(device).view(-1)
         truncated = torch.tensor(truncated).to(device).float()

         rewardBuffer[i] = reward

         #print(i, reward, done)


      #Compute the next value in the next state
      with torch.no_grad():
         _, _, _, nextValue = agent.GetActionValue(nextState)
         nextValue = nextValue.reshape(-1)

      advantage = torch.zeros_like(rewardBuffer).to(device)
      lastGAElam = torch.zeros(numEvents).to(device)

      for t in reversed(range(numSteps)):
         if t == numSteps - 1:
            # we calculated outside the loop to bootstrap.
            nextNonTerminal = 1.0 - doneBuffer[-1]
            nextValues = nextValue
         # Otherwise, we look at the next tep in the buffer
         else:
            nextNonTerminal = 1.0 - doneBuffer[t]
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


      for epoch in range(4):

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
            loss = policyLoss - 0.03*entropy.mean() + vLoss

            #Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
            optimizer.step()

      print(f"Update {step+1}/{MAX_UPDATES} | Loss: {loss.item():.4f} | Mean Reward: {rewardBuffer.sum():.2f} +- {rewardBuffer.std():.2f}")

      torch.save(agent.state_dict(), "ppo_highway_agent1.pth")



if __name__ == '__main__':

   main()