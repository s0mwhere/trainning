import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)

state_space = env.observation_space.n
action_space = env.action_space.n

#--const-------------
n_training_episodes = 10000
learning_rate = 0.7

n_eval_episodes = 100

max_steps = 99
gamma = 0.95

max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0005
#------------------

q_table = np.zeros(shape=(state_space,action_space))

for episode in range(n_training_episodes):
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    state, info = env.reset()
    step = 0
    truncated = False
    terminated = False

    for step in range(max_steps):
        random_num = np.random.uniform(0, 1)
        # exploitation
        if random_num > epsilon:
            action = np.argmax(q_table[state][:])
        # exploration
        else:
            action = env.action_space.sample()
        
        new_state, reward, terminated, truncated, info = env.step(action)
        q_table[state][action] = (1-learning_rate)*q_table[state][action] + learning_rate * (
            reward + gamma * np.max(q_table[new_state]))
        
        if terminated or truncated:
            break
        
        state = new_state

env.close()
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")


for episode in range(n_eval_episodes):
    state, info = env.reset()
    truncated = False
    terminated = False

    for step in range(max_steps):
        action = np.argmax(q_table[state][:])

        new_state, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break
        state = new_state
