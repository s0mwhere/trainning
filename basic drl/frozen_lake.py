import gymnasium as gym
import numpy as np
import random
from collections import deque
from dqn import DQN

env = gym.make("FrozenLake-v1", is_slippery=False,)
n_states = 16
n_actions = 4

# size x 1 vec for state
def one_hot(n, size):
    vec = np.zeros(size)
    vec[n] = 1
    return vec

#--const----------
episodes = 1000
max_steps = 100

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
decay_rate = 0.0005

learn_rate = 0.01
hidden_dim = 32
batch_size = 32
memory = deque(maxlen=5000)
#-----------------

dqn = DQN(n_states, hidden_dim, n_actions, learn_rate)

# Training loop
for ep in range(episodes):
    state, info = env.reset()
    total_reward = 0

    for step in range(max_steps):
        state_vec = one_hot(state, n_states)

        # action selection
        random_num = np.random.uniform(0, 1)
        if random_num < epsilon:
            action = env.action_space.sample()
        else:
            q_vals = dqn.predict(state_vec)
            action = np.argmax(q_vals)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state_vec = one_hot(next_state, n_states)

        memory.append((state_vec, action, reward, next_state_vec, done))
        state = next_state
        total_reward += reward

        #memory replay or smt
        if len(memory) >= batch_size:
            #batch = list(memory)[-batch_size:]
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = np.array(states)
            next_states = np.array(next_states)

            q_targets = dqn.predict(states)
            q_next = dqn.predict(next_states)
            max_q_next = np.max(q_next, axis=1)

            for i in range(batch_size):
                if dones[i]:
                    q_targets[i, actions[i]] = rewards[i]
                else:
                    q_targets[i, actions[i]] = rewards[i] + gamma * max_q_next[i]

            dqn.train(states, q_targets)

        if done:
            break

    # Decay epsilon
    #epsilon = max(epsilon * epsilon_decay, epsilon_min)
    epsilon = epsilon_min + (epsilon - epsilon_min) * np.exp(-decay_rate * ep)

env.close()
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

# Evaluate
for i in range(100):
    state, info = env.reset()
    for step in range(max_steps):
        state_vec = one_hot(state, n_states)
        action = np.argmax(dqn.predict(state_vec))
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
