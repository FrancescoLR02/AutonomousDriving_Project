import gymnasium
import highway_env
import numpy as np
import torch
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

from Agent import *

np.set_printoptions(linewidth=300, suppress=True, precision=5)

device = torch.device('cpu')


# Set the seed and create the environment
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

envName = "highway-fast-v0"
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "normalize": True,   
        "absolute": False,
    },

    "action":{
        "type": "DiscreteMetaAction",
        "target_speeds": [18, 21, 24, 27, 30], 
    },
    'duration': 80,
    'lanes_count': 3,
    "policy_frequency": 2,
}


env = gymnasium.make(envName, config=config, render_mode=None)

#Training hyperparameters
lr = 1e-4
gamma = 0.99
gaeLambda = 0.95
clipCoeff = 0.2

MAX_STEPS = int(1e6) 
numSteps = 1500
batchSize = 128


agent = Agent(env).to(device)
optimizer = optim.Adam(agent.parameters(), lr = lr)

stateShape = np.prod(env.observation_space.shape)

#Buffers
configSize = len(config['observation']['features'])
stateBuffer = torch.zeros((numSteps, stateShape)).to(device)
actionBuffer = torch.zeros((numSteps)).to(device)
logProbBuffer = torch.zeros((numSteps)).to(device)
rewardBuffer = torch.zeros((numSteps)).to(device)
doneBuffer = torch.zeros((numSteps)).to(device)
valuesBuffer = torch.zeros((numSteps)).to(device)

#Action mask to avoid the driver to choose illegal actions
maskBuffer = torch.zeros((numSteps, env.action_space.n), dtype = bool).to(device)


state, _ = env.reset()

#Get current state
state = torch.Tensor(state).flatten().to(device)

numEpisode = MAX_STEPS // numSteps

debug = True

successRate = []
best_reward = -float('inf')

#write rewards on file
with open('PPO/PPOrainingData.csv', 'w', newline = '') as f1:
    Data = csv.writer(f1)
    Data.writerow(['Episode', 'Avg Reward', 'SuccessRate'])

    currentEpReward = 0
    for update in range(numEpisode):
        crash = 0
        speed = []
        completedEpRewards = []

        #Replay buffer: stores numSteps 
        for i in range(numSteps):

            #Draw an action from the actor
            with torch.no_grad():
                action, logProb, _, value = agent.GetActionValue(state.unsqueeze(0))

            nextState, reward, terminated, truncated, info = env.step(action.item())

            currentEpReward += reward


            #Debug informations
            if terminated:
                crash += 1
            speed.append(info['speed'])


            done = terminated or truncated
            rewardBuffer[i] = reward
            stateBuffer[i] = state
            actionBuffer[i] = action
            logProbBuffer[i] = logProb
            doneBuffer[i] = terminated
            valuesBuffer[i] = value        
            
            if done:
                completedEpRewards.append(currentEpReward)
                successRate.append(not(info['crashed']))
                currentEpReward = 0
                nextState, _ = env.reset()

            #Update state
            state = torch.as_tensor(nextState, dtype=torch.float32, device=device).flatten()


        #Compute the advantage
        with torch.no_grad():
            _, _, _, nextValue = agent.GetActionValue(state.unsqueeze(0))
            nextValue = nextValue.item()

        #Advantage array
        advantage = torch.zeros_like(rewardBuffer).to(device)
        lastGAElam = 0

        for t in reversed(range(numSteps)):
            if t == numSteps - 1:

                #Either 0 or 1 depending whather the episode is done or not
                nextNonTerminal = 1 - done
                nextValues = nextValue

            else:
                nextNonTerminal = 1 - doneBuffer[t].item()
                nextValues = valuesBuffer[t + 1]

            #COmpute the target and advantage array 
            targetTD = rewardBuffer[t] + gamma * nextValues * nextNonTerminal - valuesBuffer[t]
            advantage[t] = lastGAElam = targetTD + gamma * gaeLambda * nextNonTerminal * lastGAElam

        returns = advantage + valuesBuffer
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

        #Training on the batch of observations
        batchState = stateBuffer.reshape((-1, stateShape))
        batchLogProb = logProbBuffer.reshape(-1)
        batchActions = actionBuffer.reshape(-1)
        batchAdvantages = advantage.reshape(-1)
        batchReturns = returns.reshape(-1)
        batchValues = valuesBuffer.reshape(-1)
        batchMask = maskBuffer.reshape((-1, env.action_space.n))

        #Use 3 times the batch to learn
        for epoch in range(6):

            idxs = np.arange(numSteps)
            np.random.shuffle(idxs)

            for start in range(0, numSteps, batchSize):
                
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
                loss = policyLoss - 0.01*entropy.mean() + 0.5*vLoss

                #Backpropagation
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                optimizer.step()



        nEpisode = int(doneBuffer.sum().item())
        if nEpisode > 0:
            avgLen = np.round(numSteps / nEpisode, 1)
        else:
            avgLen = numSteps 
        totalTerminated = nEpisode - crash
        avgSpeed = np.round(np.mean(speed), 2) if 'speed' in locals() else 0.0
        avgEpReward = np.round(np.mean(completedEpRewards), 4) if len(completedEpRewards) > 0 else 0
        avgSuccessRate = np.mean(successRate[-100:]) if len(successRate) > 0 else 0

        if update == 0:
            print(f"{'Update':<8} | {'Crashes':<8} | {'Truncated':<9} | {'avgSuccessRate':<9} | {'Avg Spd':<9} | {'Mean Reward':<12}")
            print("-" * 75)

        print(f"{update + 1}/{numEpisode:<8} | {crash:<8} | {totalTerminated:<9} | {avgSuccessRate:<9} | {avgSpeed:<9} | {avgEpReward:<12.2f}")

        if avgEpReward > best_reward:
            best_reward = avgEpReward
            torch.save(agent.state_dict(), "PPO_Champion.pth")

        Data.writerow([update, avgEpReward, avgSuccessRate])
        f1.flush()

        torch.save(agent.state_dict(), "singleTraining.pth")


    env.close()