import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
import os.path
import random
from sklearn.preprocessing import MinMaxScaler

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
    def __init__(self, data, lookback, close):
        self.data = data
        self.close = close
        self.reward = 0
        self.done = False
        self.current_step = lookback  # Start from the 'lookback'-th step
        self.inventory = 0
        self.take_profit = 0.005
        self.stop_loss = 0.003
        self.initial_price = None
        self.lookback = lookback  # The number of past steps to consider
        self.total_return = 0  # Total return
        self.potential_profit = 0  # Potential profit
        self.time_penalty = -0.0000  # Time penalty for each step
        self.transaction_cost_rate = 0.001  # Transaction cost rate
        self.num_wins = 0
        self.num_losses = 0

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data)  # Done flag updated here
        # Assign a default value to next_state
        next_state = np.zeros((self.lookback, self.data.shape[1] + 1)) # +1 for the inventory
        reward = 0

        # If it's the end of the data, we shouldn't try to access it
        if not done:
            # Buy action and not holding any stock
            if action == 1 and self.inventory == 0:
                self.inventory += 1
                self.buy_step = self.current_step  # Store the index at which the stock was bought
                self.holding_period = 0  # Reset the holding period
                reward = 0  # No immediate reward for buying

            # Check if holding stock            
            elif self.inventory > 0:
                self.holding_period += 1  # Increase the holding period
                current_price = self.close.iloc[self.current_step]['Close']
                initial_price = self.close.iloc[self.buy_step]['Close']
                percent_change = (current_price - initial_price) / initial_price

                self.potential_profit = percent_change

                # The reward is updated to reflect the unrealized profit or loss
                #reward = (self.potential_profit + (self.holding_period * self.time_penalty)) / 2

                # Apply penalty proportional to the holding period and inverse of win/loss ratio
                #reward -= self.holding_period * (1 / self.win_loss_ratio())

                # Check if take profit or stop loss is reached
                if percent_change >= self.take_profit or percent_change <= -self.stop_loss:
                    self.inventory -= 1

                    # Check if profitable
                    if percent_change >= self.take_profit:
                        self.num_wins += 1
                        #reward = (self.take_profit - self.transaction_cost_rate) * self.win_loss_ratio()
                        reward = self.take_profit - self.transaction_cost_rate
                        self.total_return += self.take_profit - self.transaction_cost_rate
                    else:
                        self.num_losses += 1
                        #reward = (-self.stop_loss - self.transaction_cost_rate) * (1 / self.win_loss_ratio())
                        reward = -self.stop_loss - self.transaction_cost_rate
                        self.total_return += -self.stop_loss - self.transaction_cost_rate

                    # Reset variables
                    self.buy_step = None  # Reset the buying step
                    self.holding_period = 0  # Reset the holding period when the position is closed
                    self.potential_profit = 0  # Reset the potential profit after selling

            next_state[:, :-1] = self.data.iloc[self.current_step - self.lookback:self.current_step].values
            next_state[:, -1] = self.inventory  # Append inventory to next_state 

        return next_state, reward, done
    
    def win_loss_ratio(self):
        if self.num_wins != 0 and self.num_losses != 0:
            return self.num_wins / (self.num_losses + 1e-9)  # To avoid division by zero
        else:
            return 1

    def reset(self):
        self.reward = 0
        self.done = False
        self.current_step = self.lookback - 1  # Start from the 'lookback'-th step
        self.total_return = 0
        self.holding_period = 0
        self.num_wins = 0
        self.num_losses = 0
        initial_state = np.zeros((self.lookback, self.data.shape[1] + 1)) # +1 for inventory
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

    def get_action(self, state, valid_actions=None):
        if valid_actions is None:
            valid_actions = [0, 1]  # Default to all actions being valid

        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)  # Randomly select from the valid actions
        else:
            state = torch.FloatTensor(state).reshape(1, -1, self.state_dim).to(device)  # Reshape state to match model input shape
            q_values = self.model(state)
            q_values_last_step = q_values[-1, 0, :]  # Get Q-values of the last step

            # Only consider Q-values of valid actions
            valid_q_values = q_values_last_step[valid_actions]
            valid_action_indexes = torch.argmax(valid_q_values).item()  # Get index of action with highest Q-value among valid actions
            return valid_actions[valid_action_indexes]

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.LongTensor(action).to(device).squeeze()  # remove extra dimensions
        reward = torch.FloatTensor(reward).to(device).squeeze()  # remove extra dimensions
        done = torch.FloatTensor(done).to(device).squeeze()  # remove extra dimensions

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

    def exploit(self, state):
        state = torch.FloatTensor(state).reshape(1, -1, self.state_dim).to(device)  # Reshape state to match model input shape
        q_values = self.model(state)
        q_values_last_step = q_values[-1, 0, :]  # Get Q-values of the last step
        return torch.argmax(q_values_last_step).item()  # Take argmax to get the action with the highest Q-value

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

def calculate_macd(data, short_window, long_window):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['Signal'] = data['MACD'].ewm(span=4, adjust=False).mean()
    data['MACD'] = data['MACD'].astype(float)
    data['Signal'] = data['Signal'].astype(float)
    return data


def calculate_rsi(data, period):
    delta = data.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    average_gain = up.rolling(window=period).mean()
    average_loss = abs(down.rolling(window=period).mean())
    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Loading OHLCV data from CSV
data = pd.read_csv('data_processing/BTC_5min_1mon_data.csv')
data = data.drop(columns='Date')
# data = data.drop( columns=[ 'Date', 'index', 'Bid1_Price', 'Bid1_Quantity', 'Ask1_Price', 'Ask1_Quantity', 
#     'Bid2_Price', 'Bid2_Quantity', 'Ask2_Price', 'Ask2_Quantity',                       
#     'Bid3_Price', 'Bid3_Quantity', 'Ask3_Price' , 'Ask3_Quantity', 
#     'Bid4_Price', 'Bid4_Quantity', 'Ask4_Price', 'Ask4_Quantity', 
#     'Bid5_Price', 'Bid5_Quantity', 'Ask5_Price', 'Ask5_Quantity'
# ])
data = data.dropna()

data['Open'] = data['Open'].astype(float)
data['High'] = data['High'].astype(float)
data['Low'] = data['Low'].astype(float)
data['Close'] = data['Close'].astype(float)
data['Volume'] = data['Volume'].astype(float)

print(data)

# Add a column to keep the original 'Close' prices
close = data[['Close']]

print(close)

# Scale the Open, High, Low, Close, and Volume columns
scaler = MinMaxScaler()
#columns_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume']
columns_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume']
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

# Split data into training, validation and test sets
train_data = data.iloc[:int(0.7*len(data))]
valid_data = data.iloc[int(0.7*len(data)):int(0.85*len(data))]
test_data = data.iloc[int(0.85*len(data)):]

# Configur pytorch to run on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the lookback period
lookback = 30

# Define the environments
train_env = TradingEnvironment(train_data, lookback, close)
valid_env = TradingEnvironment(valid_data, lookback, close)
# test_env = TradingEnvironment(test_data, lookback, close)

state_dim = 6  # Open, High, Low, Close, Volume, Inventory
action_dim = 2  # Buy, Hold
lr = 0.001
gamma = 0.99
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995

batch_size = 217 # >218 will max out cpu

# Instantiate the agent
agent = DQNAgent(state_dim, action_dim, lr, gamma, eps_start, eps_end, eps_decay)

train_output_path = 'train_output.csv'
valid_output_path = 'valid_output.csv'
test_output_path = 'test_output.csv'

# Create a variable to hold the best validation reward
best_valid_reward = float('-inf')
# Create a counter for episodes without improvement
no_improve_counter = 0
# Set a patience level - the number of episodes without improvement before stopping
patience = 10

num_episodes = 5000
for episode in range(num_episodes):
    # Training phase
    state = train_env.reset()
    while True:
        valid_actions = [0, 1] if train_env.inventory == 0 else [0]  # Only allow holding or selling if inventory is greater than 0
        action = agent.get_action(state, valid_actions=valid_actions)
        next_state, reward, done = train_env.step(action)
        agent.memory.push(state, action, reward, next_state, done)
        if train_env.current_step % 100 == 0:
            agent.update(batch_size)

        if done:
            if os.path.isfile(train_output_path):
                output = f"{episode}/{num_episodes},{train_env.total_return:.3f},{train_env.win_loss_ratio()}\n"
            else:
                output = f"episode,total_return,win_loss\n{episode}/{num_episodes},{train_env.total_return:.3f},{train_env.win_loss_ratio()}\n"
            with open(train_output_path, 'a') as f:
                f.write(output)
            break

        print(f"e: {episode}\ti: {train_env.current_step}\t\tClose: {train_env.close.iloc[train_env.current_step]['Close']}\taction: {action}\
            \tInventory: {train_env.inventory}\tPot_Profit {train_env.potential_profit:.3f}\tReward: {reward:.3f}\tReturn: {train_env.total_return:.3f}")

        state = next_state

    # Check if it's time to save the model
    if episode % 50 == 0 and episode > 99:
        # Save the model
        torch.save(agent.model.state_dict(), f'model-{episode}.pth')

    # Validation phase
    state = valid_env.reset()
    total_valid_reward = 0
    while True:
        action = agent.exploit(state)
        next_state, reward, done = valid_env.step(action)
        total_valid_reward += reward
        if done:
            if os.path.isfile(valid_output_path):
                output = f"{episode}/{num_episodes},{valid_env.total_return:.3f},{total_valid_reward:.3f}\n"
            else:
                output = f"episode, total_return,total_valid_reward\n{episode}/{num_episodes},{valid_env.total_return:.3f},{total_valid_reward:.3f}\n"
            with open(valid_output_path, 'a') as f:
                f.write(output)
            break
        state = next_state

    # Check if the validation reward is an improvement
    if total_valid_reward > best_valid_reward:
        best_valid_reward = total_valid_reward
        no_improve_counter = 0  # Reset the counter
    else:
        no_improve_counter += 1

    print(f"Episode: {episode}/{num_episodes}, Validation Total Return: {total_valid_reward:.3f}")

    # Check if patience has been exceeded
    # if no_improve_counter > patience:
    #     print("No improvement in validation reward for {0} episodes. Stopping training.".format(patience))
    #     break

# Testing phase
# state = test_env.reset()
# total_test_reward = 0
# while True:
#     action = agent.exploit(state)
#     next_state, reward, done = test_env.step(action)
#     total_test_reward += reward
#     state = next_state
#     if done:
#         if os.path.isfile(test_output_path):
#             output = f"{episode}/{num_episodes},{test_env.total_return:.3f},{total_test_reward:.3f}\n"
#         else:
#             output = f"episode, total_return,total_test_reward\n{episode}/{num_episodes},{test_env.total_return:.3f},{total_test_reward:.3f}\n"
#         with open(test_output_path, 'a') as f:
#             f.write(output)
#         break
# print(f"Test total return: {total_test_reward}")
