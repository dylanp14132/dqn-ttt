import random
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(9, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 9),
        )


    def forward(self, x):
        return self.network(x)


def masked_argmax(q_values, legal_actions): # Get the highest Q-Value from all legal actions
    mask = torch.full((9,), float("-inf"))
    mask[legal_actions] = 0.0
    return int(torch.argmax(q_values + mask))


class Agent:
    def __init__(self, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qnet = QNetwork().to(self.device)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=lr)


    def act(self, observation, legal_actions, epsilon):
        if random.random() < epsilon:
            return int(random.choice(legal_actions))
        with torch.no_grad():
            obs_t = torch.tensor(observation.flatten(), dtype=torch.float32, device=self.device)
            q = self.qnet(obs_t)
            return masked_argmax(q.cpu(), legal_actions)


    def train_step(self, observation, actions, returns):
        observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        q_values = self.qnet(observation).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.functional.mse_loss(q_values, returns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()