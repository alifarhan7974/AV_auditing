import gymnasium as gym 
import numpy as np 
import cv2 # Model requires greyscale images as input, need cv2 
import random 
from collections import deque   
  
# Torch imports     
import torch    
import torch.nn as nn  # Neural networks 
from torch.utils.tensorboard import SummaryWriter # For graphing the reward 


class DQN(nn.Module):    
    """  
    my simple implementation of a DQN   
    Takes in 4 gameplay frames, outputs Q values for best possible action   
    """
    def __init__(self, num_actions):    
        super().__init__()  
        self.conv_layers = nn.Sequential(    
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # 4 frames, 32 output channels
            nn.ReLU(),  
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # 32 frames, 64 output channels
            nn.ReLU(),  
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),  
        )

        self.fully_connected_layers = nn.Sequential(    
            nn.Flatten(),  # Takes 3d channels -> 1d vector     
            nn.Linear(64 * 7 * 7, 512),  # Paper uses 512   
            nn.ReLU(),  
            nn.Linear(512, num_actions)  # Outputs Q value for each possible action     
        )
    
    def forward(self, images):
        """
        Passes input through conv layers
        Then sends that output through FC layers to get Q-values
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

    # Added a target network to match the dqn to stay stable  
    target_network = DQN(num_actions)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = torch.optim.Adam(q_network.parameters(), lr=0.0001)

    # constant vars
    gamma = 0.99 
    epsilon = 1.0 
    batch_size = 32 
 
    # increased the size of the replay buffer
    memory = deque(maxlen=200_000) 
 
    writer = SummaryWriter("dqn_from_scratch") 

    # Training loop 
    for episode in range(episodes): 
        frame, _ = env.reset() 

        # DQN wants 4 stacked frames
        frames = deque([preprocess(frame)] * 4, maxlen=4) 
        state = np.array(frames)

        total_reward = 0 
 
        for step in range(steps): 

            # selecting action with epsilon greedy
            if random.random() < epsilon:
                action = random.randint(0, num_actions - 1)
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32).unsqueeze(0) 
                    action = q_network(s).argmax().item() 

            next_frame, reward, terminated, truncated, _ = env.step(action) 
            done = terminated or truncated 
            total_reward += reward

            # using reward clippipng to stabalize the gradients  
            clipped_reward = np.clip(reward, -1.0, 1.0) 
 
            frames.append(preprocess(next_frame)) 
            next_state = np.array(frames) 
 
            # Save transition (use clipped reward) 
            memory.append((state, action, clipped_reward, next_state, done))    

            # Only learn if enough samples exist 
            if len(memory) >= batch_size:

                batch = random.sample(memory, batch_size) 
                s, a, r, ns, d = zip(*batch) 
 
                s = torch.tensor(np.array(s), dtype=torch.float32) 
                ns = torch.tensor(np.array(ns), dtype=torch.float32) 
                a = torch.tensor(a, dtype=torch.int64) 
                r = torch.tensor(r, dtype=torch.float32)    
                d = torch.tensor(d, dtype=torch.float32)         

                # Compute Q(s, a)        
                q_values = q_network(s)          
                q_sa = q_values.gather(1, a.unsqueeze(1)).squeeze()          

                # Use target network → max Q(next_s)        
                with torch.no_grad():        
                    q_next = target_network(ns).max(1)[0]  
                    target = r + gamma * q_next * (1 - d)    

                # using hauber loss instead of mse error  
                criterion = nn.SmoothL1Loss()    
                loss = criterion(q_sa, target) 

                optimizer.zero_grad() 
                loss.backward()  
                optimizer.step()  
    
            state = next_state  
    
            # Episode ended     
            if done:    
                break   

        tau = 0.01  
        for target_param, param in zip(target_network.parameters(), q_network.parameters()):    
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)        

        epsilon = max(0.1, epsilon * 0.995)      
        writer.add_scalar("Reward per Episode", total_reward, episode)   

    writer.close() 
    env.close() 
    torch.save(q_network.state_dict(), "dqn_car_racing.pt") 


if __name__ == "__main__":
    #main(2, 10)  # quick test
    main(500, 200) 