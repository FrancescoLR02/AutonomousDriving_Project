import gymnasium
import highway_env


# Remember to save what you will need for the plots

envName = "highway-v0"
config = {
    "observation": {
        "type": "Kinematics",
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": True,   
        "absolute": False,
    },
    # "lanes_count": 3,
    # "ego_spacing": 1.5,
    "manual_control": True,
    # "policy_frequency": 5,
    # 'screen_height': 300,
    # 'screen_width': 1200,
    # 'duration': 40, 
    # 'vehicles_count': 50,
    'high_speed_reward': 12,
    # 'collision_reward': -1,

}

env = gymnasium.make(envName, config=config, render_mode='human')

env.reset()
done, truncated = False, False

episode = 1
epSteps = 0
epReturn = 0

while episode <= 10:
    epSteps += 1

    # Hint: take a look at the docs to see the difference between 'done' and 'truncated'
    obs, reward, done, truncated, _ = env.step(env.action_space.sample())  # With manual control these actions are ignored
    env.render()

    epReturn += reward

    if done or truncated:
        print(f"Episode Num: {episode} Episode T: {epSteps} Return: {epReturn:.3f}, Crash: {done}")

        env.reset()
        episode += 1
        epSteps = 0
        epReturn = 0

env.close()
