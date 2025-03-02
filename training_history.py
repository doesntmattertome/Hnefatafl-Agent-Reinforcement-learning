import pickle
import matplotlib.pyplot as plt

# Load the data from the pickle file, random file as example
with open('reward_history.pickle', 'rb') as file:
    data = pickle.load(file)

# Calculate the average percentage of rewards above 300 in the last 200 episodes
window_size = 200
average_rewards = []

for i in range(len(data)):
    if i >= window_size:
        window = data[i-window_size:i]
        rewards_above_300 = [reward for reward in window if reward > 300]
        if rewards_above_300:
            average_rewards.append(len(rewards_above_300) / window_size * 100)
        else:
            average_rewards.append(0)
    else:
        average_rewards.append(0)
print("Average % of Rewards Above 300 in the Last 200 Episodes: ", average_rewards[-1])
# Plot the results
plt.plot(average_rewards)
plt.xlabel('Episode')
plt.ylabel('Average % of Rewards Above 300')
plt.title('Average % of Rewards Above 300 in the Last 200 Episodes')
plt.show()

# Calculate the average reward in the last 200 episodes
average_reward_last_200 = []

for i in range(len(data)):
    if i >= window_size:
        window = data[i-window_size:i]
        average_reward_last_200.append(sum(window) / window_size)
    else:
        average_reward_last_200.append(sum(data[:i+1]) / (i+1))

# Plot the average reward
plt.figure()
plt.plot(average_reward_last_200)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Average Reward in the Last 200 Episodes')
plt.show()

# Plot the raw rewards
plt.figure()
plt.plot(data)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Raw Rewards')
plt.show()

# Calculate and plot the overall average reward
overall_average_reward = sum(data) / len(data)

plt.figure()
plt.plot(data, label='Raw Rewards')
plt.axhline(y=overall_average_reward, color='r', linestyle='--', label='Overall Average Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Raw Rewards with Overall Average Reward')
plt.legend()
plt.show()

# Calculate the cumulative average reward over episodes
cumulative_average_reward = [sum(data[:i+1]) / (i+1) for i in range(len(data))]

# Plot the cumulative average reward over episodes
plt.figure()
plt.plot(cumulative_average_reward, label='Cumulative Average Reward')
plt.xlabel('Episode')
plt.ylabel('Cumulative Average Reward')
plt.title('Cumulative Average Reward Over Episodes')
plt.legend()
plt.show()