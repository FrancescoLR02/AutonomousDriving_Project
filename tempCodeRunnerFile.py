config = {
    "observation": {
        "type": "Kinematics",
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": True,   
        "absolute": False,
    },
    "lanes_count": 3,
    "ego_spacing": 1.5,
    "policy_frequency": 5,
    'screen_height': 300,
    'screen_width': 1200,
    'duration': 40, 
    'vehicles_count': 50,
    'high_speed_reward': 0.8,
    'collision_reward': -5,

}