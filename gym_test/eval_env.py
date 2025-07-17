import gymnasium as gym
from stable_baselines3 import A2C, PPO
from snake_env import SnekEnv

env = SnekEnv(render_mode='human')

model_path = "gym_test/models/PPO_snake/best_model.zip"
model = PPO.load(model_path, env=env)
#model = PPO('MlpPolicy', env, verbose=0)
episodes = 50

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        print('reward', reward)

env.close()