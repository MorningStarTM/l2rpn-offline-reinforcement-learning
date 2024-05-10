import torch
import numpy as np
import grid2op
from grid2op.Action import TopologyChangeAction
from grid2op import Parameters
from grid2op.Runner import Runner
from grid2op.Episode import EpisodeData, EpisodeReplay
from agents.Q_agents import SimpleDQN, SimpleGraphDQN, OfflineQAgent



class Trainer:
    def __init__(self, cfg, n_episode):
        self.env = grid2op.make("rte_case5_example", test=True, action_class=TopologyChangeAction, p=Parameters())
        self.num_episode = n_episode
        self.losses = []
        self.scores = []
        self.average_reward_numer = []
        self.episode_numer = []
        self.best_reward = 0
        self.average_reward = 0
        self.cfg = cfg

    
    def select_agent(self, name:str):
        if name == "SimpleDQN":
            agent = SimpleDQN(self.cfg, self.env)

        elif name == "SimpleGraphDQN":
            agent = SimpleGraphDQN(self.cfg, self.env)
        
        elif name == "OfflineQAgent":
            agent = OfflineQAgent(self.cfg, self.env)
            
        return agent

    def train(self):
        for i in range(1, self.num_episode):
            state = self.env.reset()
            score = 0

            while True:
                action = self.agent.choose
