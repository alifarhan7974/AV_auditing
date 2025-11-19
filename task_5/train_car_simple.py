import gymnasium as gym
from stable_baselines3 import PPO

# 1. Create the CarRacing environment (continuous actions)
# render_mode='human' allows us to see the training 
# render_mode=None training is much faster 
env = gym.make('CarRacing-v3', render_mode=None, continuous=True)

# 2. Create the PPO model with CNN policy (since observations are images)
model = PPO('CnnPolicy', env=env, verbose=1, tensorboard_log="./car_tensorboard/")

# 3. Train the model for a short amount of timesteps (adjust for real training)
print("Training started...")
model.learn(total_timesteps=10000, tb_log_name="non_human_render")  # short for example; increase for real training

# Save model and close the env
model.save("ppo_car_model")
env.close()
