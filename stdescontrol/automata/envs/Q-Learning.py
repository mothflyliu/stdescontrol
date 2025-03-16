#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python环境：python 3.7 conda base
"""

import gym
import random
import numpy as np
import time
from policy import CustomEpsGreedyQPolicy
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'automata-v1' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]


env = gym.make('automata:automata-v1')

# env.reset(filename="automata/envs/SM/CanaisD.xml",  rewards=[-1,-1,-1,-1,2,3], stop_crit=1,last_action=60, probs=[-1,-1,-1,-1,-1,-1])
env.reset(filename="automata/envs/SM/BuffDCrash2.xml",
          rewards=[-1, -1, -1, 10, -4, -4, -1, -1], stop_crit=2, last_action=60, probs=[-1, -1, -1, -1, 0.05, 0.01, -1, -1])
# env.reset(filename="automata/envs/SM/BuffDCrash.xml",  rewards=[-1,-1,-1,10,-4,-4,-1,-1],stop_crit=1,last_action=[3], products=10, probs=[-1,-1,-1,-1,0.05,0.05,-1,-1])
num_actions = env.action_space.n
num_states = env.observation_space.n

q_table = np.zeros([num_states, num_actions], dtype=np.float32)

lr = 0.1
gamma = 0.9
epsilon = 0.1


def choose_action(env):
    pt = np.array(env.possible_transitions())
    action = pt[env.possible_space.sample()]
    return action


def epsilon_greedy(environment, epsilon, q_table, state):
    pt = np.array(environment.possible_transitions())
    uncontrollable = np.array(environment.ncontrollable)
    ptu = np.intersect1d(pt, uncontrollable)

    if(environment.probs != None and ptu.size > 0):
        probability = []
        for i in range(len(ptu)):
            if(environment.probs[ptu[i]] > 0):
                pt = np.delete(pt, np.where(pt == ptu[i]))
                probability.append(
                    [ptu[i], env.probs[ptu[i]], np.random.uniform(0, 1)])
        random.shuffle(probability)
        for i in range(len(probability)):
            if(probability[i][2] < probability[i][1]):
                return probability[i][0]
    controllable = np.array(environment.controllable)
    ptc = np.intersect1d(pt, controllable)

    if ptc.size > 0:
        if random.uniform(0, 1) < epsilon:
            action = pt[random.randint(0, pt.size-1)]
        else:
            action = ptc[np.argmax(q_table[state, ptc])]
    else:
        action = pt[random.randint(0, pt.size-1)]

    return action


'''def epsilon_greedy(env, epsilon, q_table, state): 
    action = -1
    pt = np.array(env.possible_transitions())
    uncontrollable = np.array(env.ncontrollable)
    ptu = np.intersect1d(pt,uncontrollable)
    probs = [[env.probs[ptu[i]], ptu[i]] for i in range(len(ptu)) if env.probs[ptu[i]]>0]
    
    if(len(probs)>0):
        for i in range(len(probs)):
            pt = np.delete(pt, np.where(pt==ptu[i]))
    
    while True:
    
        
        random.shuffle(probs)
        
        for i in range(len(probs)):
            if(np.random.uniform(0,1)<probs[i][0]):
                return probs[i][1]
                
        if(pt.size>0):
            controllable = np.array(env.controllable)
            ptc = np.intersect1d(pt,controllable)   
        
            if ptc.size>0:
                if random.uniform(0,1) < epsilon:
                    action = pt[random.randint(0,pt.size-1)]
                else:
                    action = ptc[np.argmax(q_table[state,ptc])]
            else: 
                    action = pt[random.randint(0,pt.size-1)]
                
        if action !=-1:
            break
        
    return action'''


def update_q_table(q_table, state, action, next_state, reward):
    old = q_table[state, action]
    next_max = np.max(q_table[next_state])

    new_value = (1-lr) * old + lr * (reward + gamma*next_max)
    q_table[state, action] = new_value


episodes = 100
start = time.time()

for i in range(episodes):
    state = env.reset()
    total_reward = 0
    step = 0
    done = False
   # env.render()

    while not done:
        step += 1
        action = epsilon_greedy(env, epsilon, q_table, state)
        #print(action)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        update_q_table(q_table, state, action, next_state, reward)
       # print("step: {}, Total Reward: {}".format(step, total_reward))
        state = next_state
        env.render()
    print("Episode: {}, Total Reward: {}".format(i+1, total_reward))


end = time.time()
print("Execution time: {}".format(end-start))
for iter_state in range(num_states):
    for iter_action in range(num_actions):
        if(q_table[iter_state, iter_action] == 0):
            q_table[iter_state, iter_action] = -100

""" episodes = 1
for i in range(episodes):
    state = env.reset()
    #env.render()
    
    done = False
    while not done:
        action = epsilon_greedy(env, epsilon, q_table, state)
        next_state, reward, done, info = env.step(action)
        update_q_table(q_table, state, action, next_state, reward)
        
        state = next_state
        env.render() """


# Testando decisões inteligentes
epsilon = 0
episodes = 5
print("Teste Inteligente")
for i in range(episodes):
    state = env.reset()
    total_reward = 0
    # env.render()

    done = False
    while not done:

        action = epsilon_greedy(env, epsilon, q_table, state)

        next_state, reward, done, info = env.step(action)
        total_reward += reward

        state = next_state
        # env.render()

    print("Episode: {}, Total Reward: {}".format(i+1, total_reward))

    # Testando decisões inteligentes
epsilon = 1
episodes = 5
print("Teste Inteligente")
for i in range(episodes):
    state = env.reset()
    total_reward = 0
    # env.render()

    done = False
    while not done:

        action = epsilon_greedy(env, epsilon, q_table, state)

        next_state, reward, done, info = env.step(action)
        total_reward += reward

        state = next_state
        # env.render()

    print("Episode: {}, Total Reward: {}".format(i+1, total_reward))
