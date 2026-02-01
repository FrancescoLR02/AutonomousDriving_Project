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
        "normalize": False,   
        "absolute": False,
    },
    "manual_control": True,
    'screen_height': 300,
    'screen_width': 1200,
    'vehicles_count': 50,
    'vehicles_density': 1.2

}

env = gymnasium.make(envName, config=config, render_mode='human')

env.reset()
done, truncated = False, False

episode = 1
epSteps = 0
epReturn = 0

files = {
    'Data': 'ManualControlActions.csv',
    'Rewards': 'ManualControlRewards.csv'
}
rewardsHeader = ['Crashed', 'Rewards']
actionsHeader = ['Speed', 'Action']

needsHeader = {key: not os.path.isfile(path) for key, path in files.items()}

#Write on file the inforations
with open(files['Data'], 'a', newline = '') as f1, open(files['Rewards'], 'a', newline = '') as f2:

    dataWriter = csv.writer(f1)
    rewardWriter = csv.writer(f2)

    #Define the headers of the csv files
    if needsHeader['Data']:
        dataWriter.writerow(actionsHeader)
    
    if needsHeader['Rewards']:
        rewardWriter.writerow(rewardsHeader)

    epReward = 0

    while True:
        
        #Take a step in the simulation
        obs, reward, done, truncated, info = env.step(env.action_space.sample())

        dataWriter.writerow([info['speed'], info['action']])

        env.render()

        #Compute final reward
        epReward += reward

        if done or truncated:
            rewardWriter.writerow([info['crashed'], epReward])
            state, _ = env.reset()

env.close()
