import gym
import random
import numpy as np
import time
from policy import CustomEpsGreedyQPolicy
from gym_cellular_automata.grid_space import GridSpace

env = gym.make("automata:automata-v0")
obs = env.reset()

total_reward = 0.0
done = False
step = 0
threshold = 12

# Random Policy for at most "threshold" steps
while not done and step < threshold:
    action = env.action_space.sample()  # Your agent goes here!
    obs, reward, done, info = env.step(action)
    total_reward += reward
    step += 1

print(f"Total Steps: {step}")
print(f"Total Reward: {total_reward}")
