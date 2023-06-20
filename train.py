import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
import os.path
import random

class Transformer(nn.Module):
    def __init__(self, state_dim, embedding_dim, num_heads, num_layers, output_dim):
        super(Transformer, self).__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.embedding = nn.Linear(state_dim, embedding_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.squeeze(dim=1)  # Remove the dimension of size 1
        x = self.fc(x)
        return x


class TradingEnvironment:
    def __init__(self, data, lookback):
        self.data = data
        self.reward = 0
        self.done = False
        self.current_step = lookback  # Start from the 'lookback'-th step
        self.inventory = 0
        self.take_profit = 1.005
        self.stop_loss = 0.997
        self.initial_price = None
        self.lookback = lookback  # The number of past steps to consider
        self.total_return = 0  # Total return
        self.potential_profit = 0  # Potential profit

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data)  # Done flag updated here
        # Assign a default value to next_state
        next_state = np.zeros((self.lookback, self.data.shape[1]))
        reward = 0

        # If it's the end of the data, we shouldn't try to access it
        if not done:
            # Buy action and not holding any stock
            if action == 1 and self.inventory == 0:
                self.inventory += 1
                self.initial_price = self.data.iloc[self.current_step]['Close']
                reward = -self.initial_price  # Negative reward as we are spending money to buy
            # Check if holding stock
            elif self.inventory > 0:
                current_price = self.data.iloc[self.current_step]['Close']
                self.potential_profit = ((current_price - self.initial_price) / self.initial_price) * 100
                # Check if take profit or stop loss is reached
                if current_price >= self.initial_price * self.take_profit or \
                current_price <= self.initial_price * self.stop_loss:
                    self.inventory -= 1  # Sell the stock
                    reward = current_price - self.initial_price
                    self.initial_price = None  # Reset the buying price
                    self.total_return += reward
                    self.potential_profit = 0  # Reset the potential profit after selling
            

            # Prepare the next state considering the past 'lookback' steps
            next_state = self.data.iloc[self.current_step - self.lookback:self.current_step].values

        return next_state, reward, done

    def reset(self):
        self.reward = 0
        self.done = False
        self.current_step = self.lookback - 1  # Start from the 'lookback'-th step
        initial_state = self.data.iloc[:self.lookback].values  # The initial state has 'lookback' steps
        self.total_return = 0
        return initial_state

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, eps_start, eps_end, eps_decay):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.model = Transformer(state_dim=state_dim, embedding_dim=state_dim, num_heads=1, num_layers=2, output_dim=action_dim).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer(10000)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 2)  # Action is now between 0 and 1
        else:
            state = torch.FloatTensor(state).reshape(1, -1, self.state_dim).to(device)  # Reshape state to match model input shape
            q_values = self.model(state)
            q_values_last_step = q_values[-1, 0, :]  # Get Q-values of the last step
            return torch.argmax(q_values_last_step).item()  # Take argmax to get the action with the highest Q-value

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.LongTensor(action).to(device).squeeze()  # remove extra dimensions
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(done).to(device)

        q_values = self.model(state)
        next_q_values = self.model(next_state)        

        q_values_last_timestep = q_values[:, -1, :]  # shape: (batch_size, num_actions)

        q_value = q_values_last_timestep.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.view(batch_size, -1).max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = self.loss_fn(q_value, expected_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.eps_end:
            self.epsilon *= self.eps_decay



class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, np.array([action]), np.array([reward]), next_state, np.array([done]))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.stack(states), np.stack(actions), np.stack(rewards), np.stack(next_states), np.stack(dones)

    def __len__(self):
        return len(self.buffer)


# Loading OHLCV data from CSV
data = pd.read_csv('BTCUSDT_data.csv')
data = data.drop( columns=[ 'Date', 'index', 'Bid1_Price', 'Bid1_Quantity', 'Ask1_Price', 'Ask1_Quantity', 
    'Bid2_Price', 'Bid2_Quantity', 'Ask2_Price', 'Ask2_Quantity',                       
    'Bid3_Price', 'Bid3_Quantity', 'Ask3_Price' , 'Ask3_Quantity', 
    'Bid4_Price', 'Bid4_Quantity', 'Ask4_Price', 'Ask4_Quantity', 
    'Bid5_Price', 'Bid5_Quantity', 'Ask5_Price', 'Ask5_Quantity'
])
data = data.dropna()

data['Open'] = data['Open'].astype(float)
data['High'] = data['High'].astype(float)
data['Low'] = data['Low'].astype(float)
data['Close'] = data['Close'].astype(float)
data['Volume'] = data['Volume'].astype(float)

print(data)

# Configur pytorch to run on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the lookback period
lookback = 30

env = TradingEnvironment(data, lookback)

state_dim = 5  # Open, High, Low, Close, Volume
action_dim = 2  # Buy, Hold
lr = 0.001
gamma = 0.99
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995

batch_size = 16

# Instantiate the agent
agent = DQNAgent(state_dim, action_dim, lr, gamma, eps_start, eps_end, eps_decay)

output_path = 'output.csv'

num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.memory.push(state, action, reward, next_state, done) # store the transition in memory
        if env.current_step % 100 == 0: # update the model every 100 steps
            agent.update(batch_size)

        if done:
            if os.path.isfile(output_path):
                output = f"{episode}/{num_episodes}, {env.total_return:.3f}\n"
            else:
                output = f"episode, total_return\n{episode}/{num_episodes}, {env.total_return:.3f}\n"
            with open(output_path, 'a') as f:
                f.write(output)
            break

        print(f"Step: {env.current_step}\tClose: {env.data.iloc[env.current_step]['Close']}\
            \tInventory: {env.inventory}\tPot_Profit {env.potential_profit:.3f}\tReturn: {env.total_return:.3f}")

        state = next_state
