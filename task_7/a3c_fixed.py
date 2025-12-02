import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter


class A3CNetwork(nn.Module):
    def __init__(self, action_dim):
        super().__init__()

        # global CNN to get features from images
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
            nn.Flatten()
        )

        # get CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 96, 96)
            conv_output_size = self.cnn(dummy_input).shape[1]

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU()
        )

        # Actor-critic architecture
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        x = x.float() / 255.0
        x = self.cnn(x)
        x = self.fc(x)

        mean_output = self.mean(x)
        std_output = torch.exp(self.log_std).expand_as(mean_output)
        value_output = self.value(x)

        return mean_output, std_output, value_output


def worker(max_global_steps, global_model, optimizer, wid, lock, log_dir):
    # every worker has own env
    env = gym.make("CarRacing-v2", continuous=True)
    local_model = A3CNetwork(action_dim=3)
    local_model.load_state_dict(global_model.state_dict())
    discount_factor = 0.99
    global_step = 0

    # Initialize TensorBoard writer for worker 0
    writer = SummaryWriter(log_dir=log_dir) if wid == 0 else None

    # initial environment state
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    while global_step < max_global_steps:
        log_probabilities = []
        value_estimates = []
        reward_list = []

        for _ in range(5):
            mean_output, std_output, value_output = local_model(state)
            distribution = torch.distributions.Normal(mean_output, std_output)
            raw_action = distribution.sample()
            action_to_environment = torch.tanh(raw_action)
            log_probability = distribution.log_prob(raw_action).sum(dim=1)

            next_state, reward, terminated, truncated, _ = env.step(action_to_environment[0].numpy())
            done = terminated or truncated

            log_probabilities.append(log_probability)
            value_estimates.append(value_output.squeeze(0))
            reward_list.append(reward)

            if done:
                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                with torch.no_grad():
                    next_value = torch.tensor(0.0)
                break
            else:
                state = torch.tensor(next_state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                with torch.no_grad():
                    _, _, next_value = local_model(state)

        # Compute returns
        R = torch.tensor(0.0) if done else next_value.detach()
        returns = []
        for r in reversed(reward_list):
            R = r + discount_factor * R
            returns.append(R)
        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float32)
        log_probabilities = torch.stack(log_probabilities)
        value_estimates = torch.stack(value_estimates)

        # Loss functions
        advantage = returns - value_estimates.detach()
        policy_loss = -(log_probabilities * advantage).mean()
        value_loss = (returns - value_estimates).pow(2).mean()
        total_loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        total_loss.backward()

        # Update global model parameters with lock
        with lock:
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                if global_param.grad is None:
                    global_param.grad = local_param.grad
                else:
                    global_param.grad.copy_(local_param.grad)
            optimizer.step()

        # Sync local model with global
        local_model.load_state_dict(global_model.state_dict())

        # Logging
        if writer is not None:
            writer.add_scalar("a3c/total_loss", total_loss.item(), global_step)
            writer.add_scalar("a3c/policy_loss", policy_loss.item(), global_step)
            writer.add_scalar("a3c/value_loss", value_loss.item(), global_step)
            writer.add_scalar("a3c/reward_chunk", sum(reward_list), global_step)

        if wid == 0:
            print(f"[Worker 0] step {global_step}, loss = {total_loss.item():.4f}")

        global_step += 1

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    lock = mp.Lock()

    global_model = A3CNetwork(action_dim=3)
    global_model.share_memory()
    optimizer = optim.Adam(global_model.parameters(), lr=1e-4)

    processes = []
    number_of_workers = 4
    max_global_steps = 10  # adjust for testing or training
    log_dir = "a3c_logs"

    for wid in range(number_of_workers):
        p = mp.Process(target=worker, args=(max_global_steps, global_model, optimizer, wid, lock, log_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()