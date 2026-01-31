import gymnasium
import highway_env
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.distributions.categorical import Categorical
from gymnasium.vector import AsyncVectorEnv

# --- 1. AGENT (No changes needed here if input is flattened correctly) ---
class Agent(nn.Module):
    def __init__(self, env_ref):
        super().__init__()
        # Auto-detect if environment is vectorized
        if hasattr(env_ref, "single_observation_space"):
            self.inputDim = np.array(env_ref.single_observation_space.shape).prod()
            self.outputDim = env_ref.single_action_space.n
        else:
            self.inputDim = np.array(env_ref.observation_space.shape).prod()
            self.outputDim = env_ref.action_space.n

        self.Network = nn.Sequential(
            nn.Linear(self.inputDim, 256),  
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.Actor = nn.Linear(256, self.outputDim)
        self.Critic = nn.Linear(256, 1)

    def GetActionValue(self, x, action=None):
        hidden = self.Network(x)
        logits = self.Actor(hidden)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        value = self.Critic(hidden)
        return action, probs.log_prob(action), probs.entropy(), value


# --- 2. CONFIGURATION ---
#device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device('cpu')

print(f"Using device: {device}")

NUM_ENVS = 8           # 8 Parallel Drivers
MAX_STEPS = int(1e6)   # 1 Million steps
STEPS_PER_ENV = 512    # Increased for better stability (Total buffer = 4096)
TOTAL_BATCH_SIZE = NUM_ENVS * STEPS_PER_ENV 

def make_env():
    def _init():
        config = {
            "observation": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy"],
                "normalize": True,
                "absolute": False, 
            },
            "lanes_count": 4,        
            "vehicles_count": 15,    
            "duration": 40,
            "policy_frequency": 5,
            
            # --- REWARD SHAPING ---
            "high_speed_reward": 0.6,
            "collision_reward": -2.0, 
            "lane_change_reward": 0.05,
            "right_lane_reward": 0.0,
        }
        return gymnasium.make("highway-v0", config=config, render_mode=None)
    return _init

# --- 3. MAIN TRAINING LOOP ---
if __name__ == "__main__":
    
    envs = AsyncVectorEnv([make_env() for _ in range(NUM_ENVS)])
    
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=5e-4)
    
    # Increase Entropy Coefficient to encourage overtaking
    entropyCoeff = 0.05 

    # Buffers 
    states = torch.zeros((STEPS_PER_ENV, NUM_ENVS, 25)).to(device)
    actions = torch.zeros((STEPS_PER_ENV, NUM_ENVS)).to(device)
    logprobs = torch.zeros((STEPS_PER_ENV, NUM_ENVS)).to(device)
    rewards = torch.zeros((STEPS_PER_ENV, NUM_ENVS)).to(device)
    dones = torch.zeros((STEPS_PER_ENV, NUM_ENVS)).to(device)
    values = torch.zeros((STEPS_PER_ENV, NUM_ENVS)).to(device)

    # --- FIX 1: Initial Flattening ---
    next_state, _ = envs.reset()
    # We flatten (8, 5, 5) -> (8, 25) so it fits the network
    next_state = torch.tensor(next_state, dtype=torch.float32).to(device).flatten(start_dim=1)
    
    num_updates = MAX_STEPS // TOTAL_BATCH_SIZE

    print(f"Starting Training on {NUM_ENVS} lanes with Total Batch Size {TOTAL_BATCH_SIZE}...")
    
    for update in range(num_updates):
        
        # --- A. Collect Trajectories ---
        for step in range(STEPS_PER_ENV):
            with torch.no_grad():
                action, logprob, _, value = agent.GetActionValue(next_state)
            
            real_next_state, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
            done = terminated | truncated

            states[step] = next_state
            actions[step] = action
            logprobs[step] = logprob
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device).view(-1)
            dones[step] = torch.tensor(done, dtype=torch.float32).to(device).view(-1)
            values[step] = value.flatten()

            # --- FIX 2: Loop Flattening ---
            # Flatten the new state immediately upon receiving it
            next_state = torch.tensor(real_next_state, dtype=torch.float32).to(device).flatten(start_dim=1)

        # --- B. Bootstrap Value (GAE) ---
        with torch.no_grad():
            _, _, _, next_value = agent.GetActionValue(next_state)
            next_value = next_value.reshape(-1)

        advantages = torch.zeros_like(rewards, dtype=torch.float32).to(device)
        lastgaelam = 0
        for t in reversed(range(STEPS_PER_ENV)):
            if t == STEPS_PER_ENV - 1:
                nextnonterminal = 1.0 - torch.tensor(done).int().to(device)
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            
            delta = rewards[t] + 0.99 * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
        
        returns = advantages + values

        # --- C. Optimize (PPO) ---
        b_states = states.reshape((-1, 25))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        b_inds = np.arange(TOTAL_BATCH_SIZE)
        
        # Train for 10 Epochs since we have a nice large batch now
        for epoch in range(10): 
            np.random.shuffle(b_inds)
            # Mini-batch size 1024
            for start in range(0, TOTAL_BATCH_SIZE, 1024): 
                end = start + 1024
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.GetActionValue(b_states[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]

                pg_loss = -mb_advantages * ratio
                pg_loss_clipped = -mb_advantages * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
                policy_loss = torch.max(pg_loss, pg_loss_clipped).mean()

                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()

                # Use higher entropy coeff to prevent "Coward" mode
                loss = policy_loss - entropyCoeff * entropy.mean() + 0.5 * v_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        if update % 5 == 0:
            print(f"Update {update}/{num_updates} | Avg Reward: {rewards.mean().item():.3f} | Best Env: {rewards.max().item():.2f}")

    torch.save(agent.state_dict(), "ppo_highway_final.pth")
    envs.close()