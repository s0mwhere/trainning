import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window_size=20):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

with open("basic drl/rewards.txt", "r") as f:
    rewards = [float(line.strip()) for line in f]

smoothed = moving_average(rewards, window_size=20)

# Plot both
plt.figure(figsize=(10, 5))
plt.plot(rewards, label="Raw Rewards", alpha=0.4)
plt.plot(np.arange(len(smoothed)) + 20 - 1, smoothed, label="Smoothed (Moving Avg)", linewidth=2)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Progress")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()