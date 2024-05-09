import torch
import numpy as np
import os
import random
from collections import namedtuple, deque


class ReplayBuffer:
    def __init__(self, MEM_SIZE, BATCH_SIZE, n_state=182):
        self.mem_count = 0
        self.MEM_SIZE = MEM_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        
        self.states = np.zeros((self.MEM_SIZE, n_state),dtype=np.float32)
        self.actions = np.zeros(self.MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(self.MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((self.MEM_SIZE, n_state),dtype=np.float32)
        self.dones = np.zeros(self.MEM_SIZE, dtype=np.bool_)
    
    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % self.MEM_SIZE
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1
    
    def sample(self):
        MEM_MAX = min(self.mem_count, self.MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, self.BATCH_SIZE, replace=True)
        
        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones
    


class GraphReplayBuffer:
    def __init__(self, MEM_SIZE, BATCH_SIZE, n_state=(21, 3), n_connection=(1,441)):
        self.mem_count = 0
        self.MEM_SIZE = MEM_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        
        self.states = np.zeros((self.MEM_SIZE, *n_state),dtype=np.float32)
        self.adj = np.zeros((self.MEM_SIZE, *n_connection))
        self.actions = np.zeros(self.MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(self.MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((self.MEM_SIZE, *n_state),dtype=np.float32)
        self.adj_ = np.zeros((self.MEM_SIZE, *n_connection),dtype=np.float32)
        self.dones = np.zeros(self.MEM_SIZE, dtype=np.bool_)
    
    def add(self, state, adj, action, reward, state_, adj_, done):
        mem_index = self.mem_count % self.MEM_SIZE
        
        self.states[mem_index]  = state
        self.adj[mem_index] = adj
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.adj_[mem_index] = adj_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1
    
    def sample(self):
        MEM_MAX = min(self.mem_count, self.MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, self.BATCH_SIZE, replace=True)
        
        states  = self.states[batch_indices]
        adj = self.adj[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        adj_ = self.adj_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, adj, actions, rewards, states_, adj_, dones