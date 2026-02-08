import gymnasium
import highway_env
import numpy as np
import torch
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": True,   
        "absolute": False,
    },
    'duration': 40,
    'lanes_count': 3,
    'initial_lane_id': None,
    "policy_frequency": 5,
    'collision_reward': -1,
    'high_speed_reward': 0.8,
    'right_lane_reward': 0.05,
    'lane_change_reward': 0.05,
    'reward_speed_range': [25, 30],
    'vehicles_count': 10,
    'vehicles_density': 1
}


env = gymnasium.make(envName, config=config, render_mode=None)

#Training hyperparameters
lr = 7e-5
gamma = 0.99
gaeLambda = 0.95
clipCoeff = 0.2

MAX_STEPS = int(5e5) 
numSteps = 1024
batchSize = 512


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

        availableActions = env.unwrapped.action_type.get_available_actions()

        nextState, reward, terminated, truncated, _ = env.step(action.item())

        rewardBuffer[i] = reward
        done = terminated or truncated
        stateBuffer[i] = state
        actionBuffer[i] = action
        logProbBuffer[i] = logProb
        doneBuffer[i] = done
        valuesBuffer[i] = value

        #print(action.item(), np.round(rewardBuffer[i], 8), done)
        
        
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

    #Use 3 times the batch to learn
    for epoch in range(3):

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
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
            optimizer.step()

    MeanReward = rewardBuffer.sum()

    print(f"Update {update+1}/{numEpisode} | Loss: {loss.item():.4f} | Mean Reward: {MeanReward:.2f}")

    torch.save(agent.state_dict(), "singleTraining.pth")


env.close()