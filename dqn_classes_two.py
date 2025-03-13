import tensorflow as tf
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, Lambda, Input, BatchNormalization,
    Activation, Add, Dropout
)
from tensorflow.keras.optimizers import Adam
import gym
from gym import spaces
import pickle
from game import Board

#dqn classes

class HnefataflEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.board = Board()
        self.action_space = spaces.Discrete(11 * 11 * 11 * 11)
        self.observation_space = spaces.Box(low=-2, high=2, shape=(11, 11), dtype=np.int32)
        self.previous_board = None
        self.previous_action = None
        self.flag = False
        self.prev_amount = 0

    def valid_moves_mask(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        valid_moves = self.board.get_possible_moves_train(self.current_player)
        for start_pos, end_positions in valid_moves.items():
            start_x, start_y = start_pos
            for end_pos in end_positions:
                end_x, end_y = end_pos
                action = (start_x * 11 * 11 * 11) + (start_y * 11 * 11) + (end_x * 11) + end_y
                mask[action] = True
        return mask

    def step(self, action):
        start_x = action // (11 * 11 * 11)
        start_y = (action % (11 * 11 * 11)) // (11 * 11)
        end_x = (action % (11 * 11)) // 11
        end_y = action % 11
        current_player = self.current_player

        self.previous_board = self.board.get_board()
        self.previous_action = [(start_x, start_y), (end_x, end_y)]

        valid = self.board.move(current_player, (start_x, start_y), (end_x, end_y))
        if (valid and (end_x, end_y) in [(10, 0), (0, 10), (10, 10), (0, 0)]):
            print("winning move and legal")

        captured = self.board.capture_enemies(current_player, (end_x, end_y))

        if not valid:
            return self.board.get_board(), -10, False, {}

        winner = self.board.check_winner()
        done = bool(winner)

        reward = 0
        if (not (-1 in self.board.get_board())) and current_player == 1 and self.flag == False:
            reward = 10
            self.flag = True
        elif captured == 2 and current_player == 1:
            reward = 3
        elif (not (-1 in self.board.get_board())) and self.flag == True:
            reward = 0.3
        amount = self.board.how_much_king_surrounded()
        if (amount < self.prev_amount):
          reward -= 12
        if (amount == 1):
          reward += 0.5
        if (amount == 2):
          reward += 1.2
        if (amount == 3):
          reward += 2
        self.prev_amount = amount


        if done:
            self.flag = False
            reward = 400 if (winner == 1 and current_player == 1) or (winner == -1 and current_player == -1) else -400

        self.current_player *= -1
        return self.board.get_board(), reward, done, {}

    def reset(self):
        self.board.reset_board()
        self.current_player = 1
        return self.board.get_board()

class DQNAgent:
    def __init__(self, state_size, action_size, player):
        self.state_size = state_size
        self.action_size = action_size
        self.player = player
        self.memory = deque(maxlen=50000)  # Default memory buffer
        self.priorities = deque(maxlen=50000)  # Priority buffer for PER
        self.gamma = 0.95
        self.epsilon = 0.99
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.tau = 0.01  # For soft updates
        self.alpha = 0.6  # PER exponent
        self.beta = 0.4   # PER importance sampling factor
        self.beta_increment = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


    def _build_model(self):
      inputs = Input(shape=(11, 11, 1))
      x = Conv2D(256, (5, 5), activation='relu', padding='same')(inputs)
      x = Conv2D(128, (5, 5), activation='relu')(x)
      x = Conv2D(64, (3, 3), activation='relu')(x)
      x = Conv2D(32, (3, 3), activation='relu')(x)
      x = Flatten()(x)

      # Dueling architecture
      advantage = Dense(512, activation='relu')(x)
      advantage = Dropout(0.5)(advantage)  # Add Dropout for regularization
      advantage = Dense(256, activation='relu')(advantage)
      advantage = Dense(self.action_size)(advantage)

      # Value Stream
      value = Dense(512, activation='relu')(x)
      value = Dropout(0.5)(value)  # Add Dropout for regularization
      value = Dense(256, activation='relu')(value)
      value = Dense(1)(value)

      # Add residual connections (skip connections) to both streams
      advantage_residual = Add()([advantage, Dense(self.action_size)(x)])
      value_residual = Add()([value, Dense(1)(x)])

      # Combine value and advantage
      # Provide an explicit output_shape
      q_values = Lambda(lambda a: a[0] + a[1] - tf.reduce_mean(a[1], axis=1, keepdims=True),output_shape=(self.action_size,)  # Explicitly specify the output shape
)([value, advantage])


      model = Model(inputs, q_values)
      model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1.0))
      return model

    def update_target_model(self):
        # Soft update the target network
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        new_weights = [
            self.tau * mw + (1 - self.tau) * tw
            for mw, tw in zip(model_weights, target_weights)
        ]
        self.target_model.set_weights(new_weights)

    def remember(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities, default=1.0)
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def _get_priority(self, error):
        return (abs(error) + 1e-5) ** self.alpha

    def replay(self, batch_size):
      if len(self.memory) < batch_size:
          return

      # Compute priorities and sample
      priorities = np.array(self.priorities)
      probabilities = priorities ** self.alpha
      probabilities /= probabilities.sum()

      indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
      minibatch = [self.memory[i] for i in indices]
      is_weights = (len(self.memory) * probabilities[indices]) ** -self.beta
      is_weights /= is_weights.max()

      # Increment beta
      self.beta = min(1.0, self.beta + self.beta_increment)

      states = np.array([experience[0] for experience in minibatch])
      actions = np.array([experience[1] for experience in minibatch])
      rewards = np.array([experience[2] for experience in minibatch])
      next_states = np.array([experience[3] for experience in minibatch])
      dones = np.array([experience[4] for experience in minibatch])

      q_values = self.model.predict(states, verbose=0)
      next_q_values_online = self.model.predict(next_states, verbose=0)
      next_q_values_target = self.target_model.predict(next_states, verbose=0)

      for i, index in enumerate(indices):
          # Ensure `target` is assigned in all cases
          target = q_values[i][actions[i]]  # Default value (should not be used in practice)
          if dones[i]:
              target = rewards[i]
          else:
              best_action = np.argmax(next_q_values_online[i])
              target = rewards[i] + self.gamma * next_q_values_target[i][best_action]

          td_error = target - q_values[i][actions[i]]
          q_values[i][actions[i]] = target

          # Update priority
          self.priorities[index] = self._get_priority(td_error)

      self.model.fit(states, q_values, sample_weight=is_weights, epochs=1, verbose=0)
      self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def act(self, state, valid_moves, exploit=False):
        if random.random() <= self.epsilon and not exploit:
            valid_indices = [i for i, valid in enumerate(valid_moves) if valid]
            return random.choice(valid_indices)

        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state, verbose=0)[0]
        q_values = np.where(valid_moves, q_values, -np.inf)
        return np.argmax(q_values)
    def model_summary(self):
        return self.model.summary()
    def print_weights(self):
        print(self.model.get_weights())
        
        