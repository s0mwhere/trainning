import gymnasium as gym
import os
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from snake_env import SnekEnv

logdir = "logs"
models_dir = "models/PPO_snake_1"                  # output model
model_path = "models/PPO_snake_1/best_model.zip"     # intput model
if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

env = SnekEnv()
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir, ent_coef=0.01)
#model = PPO.load(model_path, env=env)


'''checkpoint_callback = CheckpointCallback(
  save_freq=10000,
  save_path=models_dir,
  name_prefix="PPO"
)'''

eval_callback = EvalCallback(env, best_model_save_path=models_dir,
                             eval_freq=10000,
                             deterministic=True, render=False)

TIMESTEPS = 500000
model.learn(total_timesteps=TIMESTEPS, callback=eval_callback, tb_log_name="PPO_snake_log")

env.close()