import gymnasium
import highway_env
import numpy as np
import torch
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

from Agent import *

np.set_printoptions(linewidth=300, suppress=True, precision=5)

device = torch.device('cpu')


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
    'duration': 100,
    'lanes_count': 3,
    "policy_frequency": 2,
    #The driver starts with low crash malus 
    'collision_reward': -0.1,
    'high_speed_reward': 0.5,
    'lane_change_reward': 0.5,
    'right_lane_reward': 0,
    'reward_speed_range': [20, 30],
    'vehicles_count': 30,
    'vehicles_density': 0.8
}



env = gymnasium.make(envName, config=config, render_mode=None)

#Training hyperparameters
lr = 9e-5
gamma = 0.99
gaeLambda = 0.95
clipCoeff = 0.2

MAX_STEPS = int(5e5) 
numSteps = 1500
batchSize = 500


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

InitialMeanReward = 0
for update in range(numEpisode):
    crash = 0
    speed = []

    #Replay buffer: stores numSteps 
    for i in range(numSteps):

        #Create a mask for not available actions
        availableActions = env.unwrapped.action_type.get_available_actions()
        actionsDim = env.action_space.n
        mask = np.zeros(actionsDim, dtype = bool)
        mask[availableActions] = True
        mask = torch.as_tensor(mask, dtype = bool, device = device)

        maskBuffer[i] = mask


        #Draw an action from the actor
        with torch.no_grad():
            action, logProb, _, value = agent.GetActionValue(state.unsqueeze(0), actionMask = mask.unsqueeze(0))

            #Force exploration: choose a random action casually
            if update < numEpisode // 3 and random.random() < 0.15:
                randomAction = np.random.choice(availableActions)
                action = torch.tensor(randomAction).to(device)

        nextState, reward, terminated, truncated, info = env.step(action.item())

        #Debug informations
        if terminated:
            crash += 1
        speed.append(info['speed'])

        #Malus if the driver always choose IDLE action:
        if action.item() == 1 and speed[-1] < 25:
            reward -= 0.5

        rewardBuffer[i] = reward
        done = terminated or truncated
        stateBuffer[i] = state
        actionBuffer[i] = action
        logProbBuffer[i] = logProb
        doneBuffer[i] = done
        valuesBuffer[i] = value        
        
        if done:
            nextState, _ = env.reset()

        #Update state
        state = torch.as_tensor(nextState, dtype=torch.float32, device=device).flatten()

    #When the training is a quarter through increase collision malus
    if update == numEpisode//4:
        env.unwrapped.config['collision_reward'] = -3


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
    for epoch in range(3):

        idxs = np.arange(numSteps)
        np.random.shuffle(idxs)

        for start in range(0, numSteps, batchSize):
            
            #Select the batch indexes data
            end = start + batchSize
            idx = idxs[start : end]

            #Evaluate the batch of paris state-action
            _, newLogProb, entropy, newValue = agent.GetActionValue(batchState[idx], batchActions[idx], batchMask[idx])

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

    if debug:

        nEpisode = int(doneBuffer.sum().item())
        if nEpisode > 0:
            avgLen = np.round(numSteps / nEpisode, 1)
        else:
            avgLen = numSteps 
        totalTerminated = nEpisode - crash
        avgSpeed = np.round(np.mean(speed), 2) if 'speed' in locals() else 0.0

        if update == 0:
            print(f"{'Update':<8} | {'Crashes':<8} | {'Truncated':<9} | {'Avg Len':<9} | {'Avg Spd':<9} | {'Mean Reward':<12}")
            print("-" * 75)

        print(f"{update + 1}/{numEpisode:<8} | {crash:<8} | {totalTerminated:<9} | {avgLen:<9} | {avgSpeed:<9} | {MeanReward:<12.2f}")

    else: print(f"Update {update+1}/{numEpisode} | Loss: {loss.item():.4f} | Mean Reward: {MeanReward:.2f}")

    torch.save(agent.state_dict(), "singleTraining.pth")


env.close()