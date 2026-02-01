import gymnasium
import highway_env
import numpy as np
import torch
import random
from tqdm import tqdm

from Agent import *

np.set_printoptions(linewidth=200, suppress=True, precision=5)

device = torch.device('cpu')
print('cpu available')


# Set the seed and create the environment
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

envName = "highway-v0"
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "normalize": True,   
        "absolute": False,
    },

    "ego_spacing": 1.5,
    "policy_frequency": 5,
    'vehicles_count': 20, 
    'vehicles_density': 1.5,
    'collision_reward': -2.0,
    'lane_change_reward': 0.2,
    #'right_lane_reward': 0.0,

}


env = gymnasium.make(envName, config=config, render_mode=None)

#Training hyperparameters
lr = 9e-5
gamma = 0.99
gaeLambda = 0.95
clipCoeff = 0.2

MAX_STEPS = int(3e5) 
numSteps = 512
batchSize = 256


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

state, _ = env.reset()

#Get current state
state = torch.Tensor(state).flatten().to(device)

numEpisode = MAX_STEPS // numSteps

InitialMeanReward = 0
for update in range(numEpisode):


    #Replay buffer: stores numSteps 
    for i in range(numSteps):

        #Draw an action from the actor
        with torch.no_grad():
            action, logProb, _, value = agent.GetActionValue(state.unsqueeze(0))


        nextState, reward, terminated, truncated, _ = env.step(action.item())
        
        done = terminated or truncated

        stateBuffer[i] = state
        actionBuffer[i] = action
        logProbBuffer[i] = logProb
        rewardBuffer[i] = reward
        doneBuffer[i] = done
        valuesBuffer[i] = value

        # overtake = np.sum(np.array(nextState[1:, 1] < 0))

        # if overtake > 0:
        #     rewardBuffer[i] += overtake

        #If he doesn't crash, reward longer space travelled and more speed

        if truncated:
            rewardBuffer[i] += nextState[0, 3]*5 #+ 0.05*nextState[0, 1]

        if terminated:
            rewardBuffer[i] -= 10


        if done:
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
            nextNonTerminal = 1 - doneBuffer[t]
            nextValues = nextValue

        else:
            nextNonTerminal = 1 - doneBuffer[t + 1].item()
            nextValues = valuesBuffer[t + 1]

        #COmpute the target and advantage array 
        targetTD = rewardBuffer[t] + gamma * nextValues * nextNonTerminal - valuesBuffer[t]
        advantage[t] = lastGAElam = targetTD + gamma * gaeLambda * nextNonTerminal * lastGAElam

    returns = advantage + valuesBuffer
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    #Training on the batch of observations
    batchState = stateBuffer.reshape((-1, stateShape))
    batchLogProb = logProbBuffer.reshape(-1)
    batchActions = actionBuffer.reshape(-1)
    batchAdvantages = advantage.reshape(-1)
    batchReturns = returns.reshape(-1)
    batchValues = valuesBuffer.reshape(-1)

    #Use 3 times the batch to learn
    for epoch in range(5):

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
            loss = policyLoss - 0.03*entropy.mean() + 0.5*vLoss

            #Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    MeanReward = rewardBuffer.sum()

    if MeanReward > InitialMeanReward:
        torch.save(agent.state_dict(), "HighestReward.pth")
        InitialMeanReward = MeanReward

    print(f"Update {update+1}/{numEpisode} | Loss: {loss.item():.4f} | Mean Reward: {MeanReward:.2f}")

torch.save(agent.state_dict(), "ppo_highway_agent.pth")


env.close()