

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

import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, Lambda
from tensorflow.keras.optimizers import Adam

import pickle

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
        self.action_space = spaces.Discrete(11 * 11 * 11 * 11)
        self.observation_space = spaces.Box(low=-2, high=2, shape=(11, 11), dtype=np.int32)
        self.previous_board = None
        self.previous_action = None
        self.flag = False

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

        reward = 0.01
        if captured == 2 and current_player == 1:
            reward = 2
        if captured == 2 and self.flag == True:
            reward = 0.5
        if (not (-1 in self.board.get_board())) and current_player == 1 and self.flag == False:
            reward = 20
            self.flag = True
        if done:
            self.flag = False
            reward = 300 if (winner == 1 and current_player == 1) or (winner == -1 and current_player == -1) else -300

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
        self.memory = deque(maxlen=40000)  # Default memory buffer
        self.priorities = deque(maxlen=40000)  # Priority buffer for PER
        self.gamma = 0.95
        self.epsilon = 0.99
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.tau = 0.01  # For soft updates
        self.alpha = 0.6  # PER exponent
        self.beta = 0.4   # PER importance sampling factor
        self.beta_increment = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model = tf.keras.models.load_model(filename)
        self.target_model = tf.keras.models.load_model(filename)

    def _build_model(self):
        inputs = Input(shape=(11, 11, 1))
        x = Conv2D(128, (5, 5), activation='relu')(inputs)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = Flatten()(x)

        # Dueling architecture
        advantage = Dense(256, activation='relu')(x)
        advantage = Dense(self.action_size)(advantage)

        value = Dense(256, activation='relu')(x)
        value = Dense(1)(value)

        # Combine value and advantage
        q_values = Lambda(lambda a: a[0] + a[1] - tf.reduce_mean(a[1], axis=1, keepdims=True))([value, advantage])

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




env = HnefataflEnv()
state_size = (11, 11)
action_size = 11 * 11 * 11 * 11
load_prev = input("load previous run? (y/n): ")

black_agent = DQNAgent(state_size, action_size, 1)
black_agent.load_model('black_model.keras')
white_agent = DQNAgent(state_size, action_size, -1)

batch_size = 256
n_episodes = 12000
without_memory = 10
amount_white = 0
amount_black = 0

for episode in range(n_episodes):
    state = env.reset()
    done = False
    turn_count = 0
    wins = ""
    current_reward = 0
    while not done:
        turn_count += 1

        valid_moves = env.valid_moves_mask()
        action = black_agent.act(state, valid_moves, True)
        previous_action_black = action
        prev_state = state
        next_state, reward, done, _ = env.step(action)
        #reward_black = reward
        #black_agent.remember(state, action, reward, next_state, done)
        #state = next_state

        if done:
            #current_reward += reward
            #black_agent.remember(state, action, reward, next_state, done)
            #white_agent.remember(prev_state, previous_action_white, -reward, next_state, done)
            #print("black wins!")
            #print_board(env.previous_board)
            #print(env.previous_action)
            amount_black += 1
            wins = "black"
            break

        valid_moves = env.valid_moves_mask()
        action = white_agent.act(state, valid_moves)
        prev_state = state
        next_state, reward, done, _ = env.step(action)
        #white_agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            #current_reward += -(reward/6)
            #black_agent.remember(prev_state, previous_action_black,-(reward/6) , next_state, done)
            #print("white wins!")
            #print_board(env.previous_board)
            #print(env.previous_action)
            wins = "white"
            amount_white += 1
            break
        #else:
            #current_reward += reward_black
            #black_agent.remember(prev_state, previous_action_black, reward_black, next_state, done)
        if (turn_count % 3000 == 0):
            env.board.print_board()
            print("turn count: ", {turn_count})


    #if episode > without_memory:
    #    print("replay")
     #   black_agent.replay(batch_size)
        #white_agent.replay(batch_size)

    #if episode % 10 == 0:
    #    black_agent.update_target_model()
        #white_agent.update_target_model()
    #if episode % 40 == 0:
    #    #white_agent.save_model('white_model_two.keras')
     #   black_agent.save_model('black_model_two.keras')
    #if episode % 100 == 0:
    #    with open('black_agent_two.pickle', 'wb') as handle:
    #        pickle.dump(black_agent, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #with open('white_agent_two.pickle', 'wb') as handle:
        #    pickle.dump(white_agent, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #    with open('reward_history_black.pickle', 'wb') as handle:
    #        pickle.dump(reward_history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if episode % 1 == 0:
        print(f"Episode: {episode}, Total turns: {turn_count}, won: {wins}")
    if (episode % 100 == 0):
        print(f"amount of wins for white: {amount_white}, amount of wins for black: {amount_black}")

