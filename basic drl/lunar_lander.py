import gymnasium as gym
import numpy as np
import random
from collections import deque
from dqn import DQN

env = gym.make("LunarLander-v3")
n_states = 8
n_actions = 4

#--const-----------
episodes = 1000
max_steps = 1000

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

learn_rate = 0.001
hidden_dim=128
batch_size = 64
memory = deque(maxlen=100000)
#------------------

dqn = DQN(n_states, hidden_dim, n_actions, learn_rate)

episode_reward = []

# Training
for ep in range(episodes):
    state, info = env.reset()
    total_reward = 0

    for step in range(max_steps):
        state_input = state
        #print(state_input)
        #print(state)

        random_num = np.random.uniform(0, 1)
        if random_num < epsilon:
            action = env.action_space.sample()
        else:
            q_vals = dqn.predict(state_input)
            action = np.argmax(q_vals)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        #memory replay or smt
        if len(memory) >= batch_size:
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
    
    #logging
    episode_reward.append(total_reward)

    #epsilon decay
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    #insanity check
    if (ep + 1) % 10 == 0:
        print(f"progresssssssss!: {ep}/{episodes}")

with open("basic drl/rewards.txt", "w") as f:
    for r in episode_reward:
        f.write(f"{r}\n")

env.close()
#env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
env = gym.make("LunarLander-v3", render_mode="human")

for i in range(100):
    state, info = env.reset()
    for step in range(max_steps):
        action = np.argmax(dqn.predict(state))
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
