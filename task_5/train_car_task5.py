import gymnasium as gym
#PPO seems to be best for AV's
from stable_baselines3 import PPO

# Create enviornment
# No render mode will make program run faster
env = gym.make('CarRacing-v2', render_mode=None, continuous=True)

# Create model, CnnPolicy is best
model = PPO('CnnPolicy', env=env, verbose=1, tensorboard_log="./car_tensorboard/")

# Model is training
model.learn(total_timesteps=100000, tb_log_name="100k_timesteps")

# Save model and close
model.save("farhans_car_model")
env.close()