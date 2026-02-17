import numpy as np
import gymnasium
import highway_env
import csv
import os

np.set_printoptions(linewidth=200, suppress=True, precision=5)


# Remember to save what you will need for the plots

envName = "highway-v0"
config = {
    "observation": {
        "type": "Kinematics",
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": True,   
        "absolute": False,
    },
    "manual_control": True,
    'screen_height': 300,
    'screen_width': 1200,
    'duration': 80,
    "policy_frequency": 2


}

env = gymnasium.make(envName, config=config, render_mode='human')

env.reset()
done, truncated = False, False

episode = 1
epSteps = 0
epReturn = 0

files = {
    'Data': 'Data/ManualControlActions.csv',
    'Rewards': 'Data/ManualControlRewards.csv'
}
rewardsHeader = ['Crashed', 'Rewards', 'AvgSpeed', 'StdSpeed']
actionsHeader = ['Speed', 'Action']

needsHeader = {key: not os.path.isfile(path) for key, path in files.items()}

avgSpeed = []

#Write on file the inforations
with open(files['Data'], 'a', newline = '') as f1, open(files['Rewards'], 'a', newline = '') as f2:

    #dataWriter = csv.writer(f1)
    rewardWriter = csv.writer(f2)

    #Define the headers of the csv files
    # if needsHeader['Data']:
    #     dataWriter.writerow(actionsHeader)
    
    if needsHeader['Rewards']:
        rewardWriter.writerow(rewardsHeader)

    epReward = 0

    run = True
    while run:
        
        #Take a step in the simulation
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
        #print(info['action'], np.round(reward, 4))
        #dataWriter.writerow([info['speed'], info['action']])
        avgSpeed.append(info['speed'])

        env.render()

        #Compute final reward
        epReward += reward

        if done or truncated:
            rewardWriter.writerow([info['crashed'], epReward, np.mean(avgSpeed), np.std(avgSpeed)])
            epReward = 0
            avgSpeed = []
            state, _ = env.reset()
            f2.flush()


env.close()
