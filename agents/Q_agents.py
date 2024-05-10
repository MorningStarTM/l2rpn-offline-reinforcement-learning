import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np
import pickle
from utils.replay_buffer import ReplayBuffer, GraphReplayBuffer
from agents.network import QNetwork, GraphNetwork
from utils.converter import Converter
from utils.node import Node


class SimpleDQN:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.memory = ReplayBuffer(self.cfg['MEM_SIZE'], self.cfg['BATCH_SIZE'])
        self.cfg = cfg
        self.env = env
        self.exploration_rate = self.cfg['EXPLORATION_MAX']
        self.network = QNetwork(self.cfg)
        self.losses = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def choose_action(self, observation):
        if random.random() < self.exploration_rate:
            return self.env.action_space.sample()
        
        state = torch.tensor(observation).float().detach()
        state = state.to(self.device)
        state = state.unsqueeze(0)
        q_values = self.network(state)
        return torch.argmax(q_values).item()
    
    def learn(self):
        if self.memory.mem_count < self.cfg['BATCH_SIZE']:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample()
        states = torch.tensor(states , dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        states_ = torch.tensor(states_, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        batch_indices = np.arange(self.cfg['BATCH_SIZE'], dtype=np.int64)

        q_values = self.network(states)
        next_q_values = self.network(states_)
        
        predicted_value_of_now = q_values[batch_indices, actions]
        predicted_value_of_future = torch.max(next_q_values, dim=1)[0]
        
        q_target = rewards + self.cfg['GAMMA'] * predicted_value_of_future * dones

        loss = self.network.loss(q_target, predicted_value_of_now)
        self.losses.append(loss)
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()

        self.exploration_rate *= self.cfg['EXPLORATION_DECAY']
        self.exploration_rate = max(self.cfg['EXPLORATION_MIN'], self.exploration_rate)

    def returning_epsilon(self):
        return self.exploration_rate
    
    def save_model(self, path):
        torch.save(self.network.state_dict(), path)

    def load_model(self, path):
        self.network.load_state_dict(path)






class SimpleGraphDQN:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.memory = GraphReplayBuffer(self.cfg['MEM_SIZE'], self.cfg['BATCH_SIZE'])
        self.env = env
        self.converter = Converter(self.env)
        self.exploration_rate = self.cfg['EXPLORATION_MAX']
        self.network = GraphNetwork(self.cfg['node_feature'], self.cfg['action_dim'])
        self.losses = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def choose_action(self, observation, adj):
        if random.random() < self.exploration_rate:
            return self.converter.convert_env_act_to_one_hot_encoding_act(self.env.action_space.sample().to_vect())
        
        
        q_values = self.network(observation, adj)
        return torch.argmax(q_values).item()
    
    def learn(self):
        if self.memory.mem_count < self.cfg['BATCH_SIZE']:
            return
        
        states, adj, actions, rewards, states_, adj_, dones = self.memory.sample()
        states = torch.tensor(states , dtype=torch.float32).to(self.device)
        adj = torch.tensor(adj , dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        states_ = torch.tensor(states_, dtype=torch.float32).to(self.device)
        adj_ = torch.tensor(adj_ , dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        batch_indices = np.arange(self.cfg['BATCH_SIZE'], dtype=np.int64)

        q_values = self.network(states, adj)
        next_q_values = self.network(states_, adj_)
        
        predicted_value_of_now = q_values[batch_indices, actions]
        predicted_value_of_future = torch.max(next_q_values, dim=1)[0]
        
        q_target = rewards + self.cfg['GAMMA'] * predicted_value_of_future * dones

        loss = self.network.loss(q_target, predicted_value_of_now)
        self.losses.append(loss)
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()

        self.exploration_rate *= self.cfg['EXPLORATION_DECAY']
        self.exploration_rate = max(self.cfg['EXPLORATION_MIN'], self.exploration_rate)

    def returning_epsilon(self):
        return self.exploration_rate
    
    def save_model(self, path):
        torch.save(self.network.state_dict(), path)

    def load_model(self, path):
        self.network.load_state_dict(path)

##### offline Q agents

class OfflineQAgent:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.converter = Converter(self.env)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = GraphReplayBuffer(self.cfg['MEM_SIZE'], self.cfg['BATCH_SIZE']) 
        self.q_net = QNetwork(self.cfg['observation_dim'], self.cfg['action_dim']).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.cfg['LR'])


    def choose_action(self, state):
        if random.random > self.cfg['EPSILON_RATE']:
            return self.converter.convert_env_act_to_one_hot_encoding_act(self.env.action_space.sample().to_vect())
        
        
        q_val = self.q_net(state)
        return torch.argmax(q_val).item()
    

    def learn(self, states, actions, rewards, dones):
        states = torch.tensor(states , dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        states_ = torch.tensor(states_, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        batch_indices = np.arange(self.cfg['BATCH_SIZE'], dtype=np.int64)

        q_values = self.network(states)
        next_q_values = self.network(states_)
        
        predicted_value_of_now = q_values[batch_indices, actions]
        predicted_value_of_future = torch.max(next_q_values, dim=1)[0]
        
        q_target = rewards + self.cfg['GAMMA'] * predicted_value_of_future * dones

        loss = self.network.loss(q_target, predicted_value_of_now)
        self.losses.append(loss)
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()

        self.exploration_rate *= self.cfg['EXPLORATION_DECAY']
        self.exploration_rate = max(self.cfg['EXPLORATION_MIN'], self.exploration_rate)


    def returning_epsilon(self):
        return self.exploration_rate
    
    def save_model(self, path):
        torch.save(self.network.state_dict(), path)

    def load_model(self, path):
        self.network.load_state_dict(path)


        