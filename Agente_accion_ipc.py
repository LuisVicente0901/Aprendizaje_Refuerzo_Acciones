# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 16:24:09 2022

@author: Luis Vicente
"""

import gym
import gym_anytrading

# Stable baselines - rl stuff
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, ACKTR

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time



df = pd.read_csv('C:/Users/Luis Vicente/Documents/Tesis/Proyecto Tesis/Prediccion-ML/datos_A_Movil.csv')

df['Date'] = pd.to_datetime(df['Date'])
df.dtypes

df.set_index('Date', inplace=True)
df.head()

env = gym.make('stocks-v0', df=df,frame_bound=(1600,1700), window_size=5)
env.signal_features

env.action_space

env.action_space.seed(0)
state = env.reset()
while True: 
    action = env.action_space.sample()
    n_state, reward, done, info = env.step(action)
    if done: 
        print("info", info)
        break
        
plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()


env_maker = lambda: gym.make('stocks-v0', df=df ,frame_bound=(1600,1720), window_size=5)
env = DummyVecEnv([env_maker])

model = A2C('MlpLstmPolicy', env, verbose=1,seed=0) 
model.learn(total_timesteps=5500)


env = gym.make('stocks-v0', df=df, frame_bound=(1721,1751), window_size=5)
obs = env.reset()

while True: 
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        break
    

#MODIFICAR CICLO PARA OBTENER UN RENDIMIENTO POSITIVO

start = time.time()
info={'total_reward': 0,'total_profit': 0,'position': 0}
while info['total_profit']<1.08:
    env = gym.make('stocks-v0', df=df, frame_bound=(1721,1751), window_size=5)
    obs = env.reset()
    while True: 
        obs = obs[np.newaxis, ...]
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            break
print("info", info)
print("it took", time.time() - start, "seconds.")   

plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()




#########################################################################
#MEJORAR LOS RESULTADOS USANDO MÁS VARIABLES
from gym_anytrading.envs import StocksEnv

def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Close'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Close', 'Open', 'High', 'Low']].to_numpy()[start:end]
    return prices, signal_features

class MyCustomEnv(StocksEnv):
    _process_data = my_process_data
    
env2 = MyCustomEnv(df=df, window_size=5, frame_bound=(1600,1720))

env2.signal_features


env_maker = lambda: env2
env = DummyVecEnv([env_maker])

model = A2C('MlpLstmPolicy', env, verbose=1,seed=0) 
model.learn(total_timesteps=5500)


env = MyCustomEnv(df=df, window_size=5, frame_bound=(1721,1751))
obs = env.reset()
while True: 
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        break
    
plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()


start = time.time()
info={'total_reward': 0,'total_profit': 0,'position': 0}
while info['total_profit']<1.08:
    env = MyCustomEnv(df=df, window_size=5, frame_bound=(1721,1751))
    obs = env.reset()
    while True: 
        obs = obs[np.newaxis, ...]
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            break
print("info", info)
print ("it took", time.time() - start, "seconds.")

plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()

######################################################################
#PROBAR SOLO USANDO LOS VALORES CLOSE

from gym_anytrading.envs import StocksEnv

def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Close'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Close']].to_numpy()[start:end]
    return prices, signal_features

class MyCustomEnv(StocksEnv):
    _process_data = my_process_data
    
env2 = MyCustomEnv(df=df, window_size=5, frame_bound=(1600,1720))

env2.signal_features


env_maker = lambda: env2
env = DummyVecEnv([env_maker])

model = A2C('MlpLstmPolicy', env, verbose=1,seed=0) 
model.learn(total_timesteps=5500)


env = MyCustomEnv(df=df, window_size=5, frame_bound=(1721,1751))
obs = env.reset()
while True: 
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        break


info={'total_reward': 0,'total_profit': 0,'position': 0}
while info['total_profit']<1.08:
    env = MyCustomEnv(df=df, window_size=5, frame_bound=(1721,1751))
    obs = env.reset()
    while True: 
        obs = obs[np.newaxis, ...]
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            break
print("info", info)

plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()



start = time.time()
info={'total_reward': 0,'total_profit': 0,'position': 0}
while info['total_profit']<1.08:
    env = MyCustomEnv(df=df, window_size=5, frame_bound=(1721,1751))
    obs = env.reset()
    while True: 
        obs = obs[np.newaxis, ...]
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            break
print("info", info)
print ("it took", time.time() - start, "seconds.")

##########################################################################
#USANDO ACER


env_maker = lambda: gym.make('stocks-v0', df=df ,frame_bound=(1600,1720), window_size=5)
env = DummyVecEnv([env_maker])

model = ACKTR('MlpLstmPolicy', env, verbose=1,seed=0) 
model.learn(total_timesteps=5500)


env = gym.make('stocks-v0', df=df, frame_bound=(1721,1751), window_size=5)
obs = env.reset()


#MODIFICAR CICLO PARA OBTENER UN RENDIMIENTO POSITIVO

start = time.time()
info={'total_reward': 0,'total_profit': 0,'position': 0}
while info['total_profit']<1.1:
    env = gym.make('stocks-v0', df=df, frame_bound=(1721,1751), window_size=5)
    obs = env.reset()
    while True: 
        obs = obs[np.newaxis, ...]
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            break
print("info", info)
print("it took", time.time() - start, "seconds.")   

plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()
