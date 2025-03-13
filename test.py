import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
import gym
from gym import spaces
from game import Board

import pickle

# function from board:
def print_board(board, positions_marked = []):
        column_index = 0
        for column in board:
            row_index = 0
            if (column_index < 10):
                if (column_index != 0):
                    print(column_index, end="  ")
            else:
                print(column_index, end=" ")
            for row in column:
                if (column_index == 0 and row_index == 0):
                    print("   ", end = "")
                    for i in range(11):
                        print(i, end = " ")
                    print("")
                    print("0  ", end="")                    
                if (row_index, column_index) in positions_marked:
                    print("X", end = " ")
                elif row == 0:
                    print("▢", end = " ")
                elif row == 1:
                    print("B", end = " ")
                elif row == -1:
                    print("W", end = " ")
                elif row == -2:
                    print("K", end = " ")
                elif row == 2:
                    print("▨", end = " ")
                
                
                row_index += 1
            column_index += 1
            row_index = 0
            print()


class HnefataflEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.board = Board()
        self.action_space = spaces.Discrete(11 * 11 * 11 * 11)  # All possible moves from-to
        self.observation_space = spaces.Box(low=-2, high=2, shape=(11, 11), dtype=np.int32)
        self.previous_board = None
        self.previous_action = None
        
    def valid_moves_mask(self):
        """Generate a binary mask for all possible actions."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        valid_moves = self.board.get_possible_moves_train(self.current_player)  # Returns a dict
        for start_pos, end_positions in valid_moves.items():
            start_x, start_y = start_pos
            for end_pos in end_positions:
                end_x, end_y = end_pos
                action = (start_x * 11 * 11 * 11) + (start_y * 11 * 11) + (end_x * 11) + end_y
                mask[action] = True
        return mask

    def step(self, action):
        # Convert action number to board coordinates
        start_x = action // (11 * 11 * 11)
        start_y = (action % (11 * 11 * 11)) // (11 * 11)
        end_x = (action % (11 * 11)) // 11
        end_y = action % 11
        current_player = self.current_player
        
        self.previous_board = self.board.get_board()
        self.previous_action = [(start_x, start_y), (end_x, end_y)]
        
        valid = self.board.move(current_player, (start_x, start_y), (end_x, end_y))
        #if (valid and (end_x, end_y) in [(10, 0), (0, 10), (10, 10), (0, 0)]):
        #    print("winning move and legal")
        
        captured = self.board.capture_enemies(current_player, (end_x, end_y))
        
        if not valid:
            return self.board.get_board(), -10, False, {}  # Penalize illegal moves less aggressively
            
        winner = self.board.check_winner()
        done = bool(winner) 
        
        
        reward = 0.005
        if captured == 2 and current_player == 1:
            reward = 2
        # check if there are any white pieces left
        if (not (-1 in self.board.get_board())) and current_player == 1:
            reward = 20
        if done:
            reward = 100 if (winner == 1 and current_player == 1) or (winner == -1 and current_player == -1) else -100

        self.current_player *= -1  # Switch players
        #print("current player: ", self.current_player)
        return self.board.get_board(), reward, done, {}
    
    def reset(self):
        self.board.reset_board()
        self.current_player = 1  # Black starts
        return self.board.get_board()


class DQNAgent:
    def __init__(self, state_size, action_size, player):
        self.state_size = state_size
        self.action_size = action_size
        self.player = player
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 1
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model = tf.keras.models.load_model(filename)
        self.target_model = tf.keras.models.load_model(filename)
    
    def _build_model(self):
        model = Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(11, 11, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(11, 11, 1)),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_moves, exploit=False):
        """Choose an action, masking invalid moves."""
    
        if random.random() <= self.epsilon and not exploit:
            valid_indices = [i for i, valid in enumerate(valid_moves) if valid]
            
            return random.choice(valid_indices)

        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state, verbose=0)[0]
        q_values = np.where(valid_moves, q_values, -np.inf)  # Mask invalid moves
        return np.argmax(q_values)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.target_model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0]
                )
            target_f = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
            target_f[0][action] = target
            states.append(state)
            targets.append(target_f[0])
        
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Training setup
env = HnefataflEnv()
state_size = (11, 11)
action_size = 11 * 11 * 11 * 11

black_agent = DQNAgent(state_size, action_size, 1)
white_agent = DQNAgent(state_size, action_size, -1)

print("start")
batch_size = 64
n_episodes = 12_000
without_memory = 10
white_wins = 0
black_wins = 0
# Training loop
for episode in range(n_episodes):
    state = env.reset()
    done = False
    turn_count = 0  # Initialize turn counter for this episode

    previous_action_white = None
    previous_action_black = None
    while not done:
        turn_count += 1  # Increment counter for each turn

        # Black's turn
        valid_moves = env.valid_moves_mask()
        action = black_agent.act(state, valid_moves)
        prev_state = state
        next_state, reward, done, _ = env.step(action)
        state = next_state
    
        if done:
            black_wins += 1
            break
        # White's turn
        
        valid_moves = env.valid_moves_mask()
        action = white_agent.act(state, valid_moves)
        prev_state = state
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if (done):
            white_wins += 1
            break
        previous_action_black = action
        previous_action_white = action
        
    
    if episode % 100 == 0:
        print(f"white won: {white_wins}, black won: {black_wins}. this is episode: {episode}")
        
    # Log progress
    if episode % 1 == 0:
        print(f"Episode: {episode}, Total turns: {turn_count}, Epsilon: {black_agent.epsilon}")


