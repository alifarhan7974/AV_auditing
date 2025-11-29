import gymnasium as gym
import numpy as np
import cv2 # Model requires greyscale images as input, need cv2
import random
from collections import deque

# Torch imports
import torch
import torch.nn as nn # Neural networks
from torch.utils.tensorboard import SummaryWriter # For graphing the rewaard

class DQN(nn.Module):
    """
    my siimple implementation of a DQN
    Takes in 4 gameplay frames, outputs Q values for best possible action
    """
    def __init__(self, num_actions):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), # 4 frames, 32 output channels
            nn.ReLU(),  # Filters out the negative vals
            nn.Conv2d(32, 64, kernel_size = 4, stride=2), # 32 frames, 64 ouptut channels
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fully_connected_layers = nn.Sequential(
            # Decides what action to take based on conv_layers
            nn.Flatten(), # Takes 3d channels -> 1d vector
            nn.Linear(64 * 7 * 7, 512),  # Paper uses 512
            nn.ReLU(),
            nn.Linear(512, num_actions)  # Outputs Q value for each possible action
        )

    def forward(self, images):
        """
        Passes input through conv_layers
        Then sends that outupt thru fc layers to get output Q_vals

        """
        return self.fully_connected_layers(self.conv_layers(images))


def preprocess(frame):
    """
    Converts raw images to understandable input
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))
    frame = frame.astype(np.float32) / 255.0
    return frame


def main(steps, episodes):
    env = gym.make("CarRacing-v2", continuous=False)
    num_actions = env.action_space.n
    q_network = DQN(num_actions)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=0.0001)

    # Constant vars
    gamma = 0.99
    epsilon = 1.0
    batch_size = 32

    # Replay buffer to store past experiences
    memory = deque(maxlen=50000)

    # For graphing with tensorboard
    writer = SummaryWriter("dqn_from_scratch")

    # Training loop
    for episode in range(episodes): # Episode is one training cycle of rl agent

        frame, _ = env.reset()
        #DQN wants 4 stacked frames
        frames = deque([preprocess(frame)] * 4, maxlen=4)
        state = np.array(frames)

        # Graphing the reward with tensorboard
        total_reward = 0

        # Trainin for 1000 steps
        for step in range(steps):

            # Selecting action with epislon-greedy
            if random.random() < epsilon:
            # Do something random
                action = random.randint(0, num_actions - 1)
            else:
            # Choose optimal Q val
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action = q_network(s).argmax().item()

            # Get next frames and check if episode is done
            next_frame, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated # Boolean trick to make cut lines
            total_reward += reward

            # Push newest frame to queue, pop the oldest
            frames.append(preprocess(next_frame))
            next_state = np.array(frames)

            # Save transition
            memory.append((state, action, reward, next_state, done))

            if len(memory) >= batch_size:
                # Only learn from the memory if its greater than batchsize
                batch = random.sample(memory, batch_size)
                s, a, r, ns, d = zip(*batch)

                s = torch.tensor(np.array(s), dtype=torch.float32)
                ns = torch.tensor(np.array(ns), dtype=torch.float32)
                a = torch.tensor(a, dtype=torch.int64)
                r = torch.tensor(r, dtype=torch.float32)
                d = torch.tensor(d, dtype=torch.float32)

                # Compute Q(s,a)
                q_values = q_network(s)
                q_sa = q_values.gather(1, a.unsqueeze(1)).squeeze()

                # Compute target = r + gamma * max(Q(next_s))
                # Bellman eqn
                with torch.no_grad():   # context manager
                    q_next = q_network(ns).max(1)[0]
                    target = r + gamma * q_next * (1 - d)

                loss = ((target - q_sa) ** 2).mean() # Get loss and update nn
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state

            if done:
                break

        epsilon = max(0.1, epsilon * 0.995) 
        writer.add_scalar("Reward per Episode", total_reward, episode) 
        #print(f"Episode {episode}, Reward: {total_reward}") # Debugging outpout 

    writer.close()
    env.close()
    torch.save(q_network.state_dict(), "dqn_car_racing.pt")

if __name__ == "__main__": 
    #main(2, 100) # Testing
    main(500, 200) 

