# Autonomous Driving with Reinforcement Learning

## hierarchy and How to Run:

Only the models, scripts and Analysis jupyter are given. The data gathered is only visualized in the Jupter Notebook.

In the three main folders are presents the scripts to run the training of the models (PPO, DQN and CL framework) (PPO also present a parallelized version of the algorithm, but this was at least not used.)

The optimal weights for each model are respectively ...

### Baseline: 

run:
```bash
python PPO_evaluate.py True
```
("True" runs the baseline)

### PPO model:
run:
```bash
python PPO_evaluate.py
```

### DQN model:
run:
```bash
python DQN_evaluate.py
```

...






## Presentation of the project 

This project explores the application of Reinforcement Learning (RL) algorithms to train autonomous agents in complex, heterogeneous highway environments. It compares the performance of a Deep Q-Network (DQN) and Proximity Policy Optimization (PPO) against a heuristic baseline policy using the `highway-env` gymnasium framework. 

Additionally, the project evaluates the models' adaptability through a Continual Learning (CL) setup, transferring agents from a standard highway navigation task to a complex merging scenario. 

## Environment Setup

The driving environment is modeled as a Markov Decision Process (MDP) with the following characteristics:

* **State Space:** A 10x7 matrix tracking the 10 closest vehicles. Features include a boolean presence bit, spatial positions $(x, y)$, velocities $(v_x, v_y)$, and angular positions ($\cosh$, $\sinh$). The input is flattened into a 1D vector for the neural networks.
* **Action Space:** A discrete meta-action space consisting of 5 actions: `[Faster, Slower, Right Lane, Left Lane, Idle]`.
* **Reward Function:** Customized to encourage confident overtaking and speed. The default reward for remaining in the rightmost lane is set to 0 to prevent over-conservative policies. It includes a bonus for reaching high speeds and a penalty for collisions.

## Model Architectures

To ensure a fair comparison, both RL agents utilize a similar core neural network structure.

### 1. Deep Q-Network (DQN)
An off-policy algorithm that uses a Replay Buffer to improve sample efficiency and random $\varepsilon$-greedy exploration.
* **Architecture:** Feed-Forward Neural Network (FFNN) with two fully connected hidden layers containing 128 and 256 neurons. It uses ReLU activation for hidden layers and a linear output layer for the 5 discrete Q-values.
* **Optimization:** Trained using Huber Loss (to handle temporal difference error outliers) and the Adam optimizer (Learning Rate: 1e-4, Discount Factor: 0.99). 
* **Exploration:** $\varepsilon$ decays from 0.9 to 0.01 over 25,000 steps.

### 2. Proximity Policy Optimization (PPO)
An on-policy Actor-Critic algorithm that optimizes the target policy directly while ensuring updates do not deviate excessively from the old policy.
* **Architecture:** Both Actor and Critic networks use the identical FFNN architecture as the DQN (128, 256 neurons, ReLU). 
* **Output:** The Actor outputs a probability distribution over the discrete action space, while the Critic outputs a scalar representing the expected cumulative reward.
* **Optimization:** Uses a clipped objective function (clip coefficient: 0.2), alongside entropy and state-value loss terms.

### 3. Baseline Policy
A heuristic, non-ML baseline agent that makes decisions using hardcoded positional and velocity thresholds. It identifies the closest vehicles and executes basic spatial rules to determine lane changes and acceleration.

## Project Structure

* `trainingDQN.py`: Handles the setup, `highway-v0` environment configuration, and training loop for the off-policy DQN agent.
* `modelDQN.py`: Contains the PyTorch neural network class definitions.
* `ReplayBuffer.py`: Manages the storage and sampling of state transitions for the DQN.
* `Analysis.ipynb`: Evaluates the trained agents by analyzing CSV logs, plotting speed distributions, and comparing success metrics between the RL agents and the baseline.