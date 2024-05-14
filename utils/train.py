import torch
import numpy as np
import grid2op
from grid2op.Action import TopologyChangeAction
from grid2op import Parameters
from grid2op.Runner import Runner
from grid2op.Episode import EpisodeData, EpisodeReplay
from agents.Q_agents import SimpleDQN, SimpleGraphDQN, OfflineQAgent
from utils.converter import Converter
from utils.node import Node
import matplotlib.pyplot as plt



class Trainer:
    def __init__(self, cfg):
        self.env = grid2op.make("rte_case5_example", test=True, action_class=TopologyChangeAction)
        self.losses = []
        self.scores = []
        self.average_reward_numer = []
        self.episode_numer = []
        self.best_reward = 0
        self.average_reward = 0
        self.cfg = cfg
        self.converter = Converter(self.env)
    
    def select_agent(self, name:str):
        if name == "SimpleDQN":
            agent = SimpleDQN(self.cfg, self.env)

        elif name == "SimpleGraphDQN":
            agent = SimpleGraphDQN(self.cfg, self.env)
        
        elif name == "OfflineQAgent":
            agent = OfflineQAgent(self.cfg, self.env)

        return agent

    def train_on_DQN(self, n_episode):
        agent = self.select_agent("SimpleDQN")
        for i in range(1, n_episode):
            state = self.env.reset()
            score = 0

            while True:
                action = agent.choose_action(state.to_vect())
                new_state, reward, done, _ = self.env.step(self.converter.convert_one_hot_encoding_act_to_env_act(self.converter.int_one_hot(action)))
                print(action)
                agent.memory.add(state.to_vect(), action, reward, new_state.to_vect(), done)
                agent.learn()
                state = new_state
                score += reward

                if done:
                    if score > self.best_reward:
                        self.best_reward = score
                    self.average_reward += score
                    print("Episode {} Average Reward {} Best Reward {} Last Reward {} Epsilon {}".format(i, self.average_reward/i, self.best_reward, score, agent.returning_epsilon()))
                    break

                self.episode_numer.append(i)
                self.average_reward_numer.append(self.average_reward / i)

    def train_non_graph_based_agent(self, name, trajectory_data, n_episode):
        agent = self.select_agent(name)
        for i in range(n_episode):
            for batch in trajectory_data:
                states, actions, rewards, next_states, dones = batch
                agent.learn(states, actions, rewards, next_states, dones)
            
            self.scores = []
            mean_rewards = []
            counter = 0


            while counter < self.cfg['EVAL_EPISODES']:
                action = agent.act(states)
                state, reward, done, _ = self.env.step(self.converter.convert_one_hot_encoding_act_to_env_act(action))
                self.scores.append(reward)
                if done:
                    self.env.reset()
                    mean_rewards.append(sum(rewards))
                    rewards = []
                    counter += 1

    
    def plot_learning(self):
        fig, ax = plt.subplots(figsize=(10, 5))  # Rectangle shape

        # Plot the losses
        ax.plot(self.average_reward_number, color='blue', alpha=0.6, label='Loss')

        # Customize the plot
        ax.set_title('DQN Losses')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Reward')
        ax.legend()
        plt.savefig('result\\DQN_reward')

