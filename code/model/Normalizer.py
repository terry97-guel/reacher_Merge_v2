# %%
import torch
from torch import Tensor

import torch
from torch.nn import Module
from copy import deepcopy as dc

class RunningNormalizer(Module):
    def __init__(self,decay=0.95):
        super().__init__()
        self.register_buffer("Running_mean", torch.zeros(4).float())
        self.num_updates = 0
        self.decay = decay
    def update(self, rewards, quadrants):
        Running_mean = self.get_buffer("Running_mean")
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(
                self.decay,
                (1 + self.num_updates) / (10 + self.num_updates)
            )
            
        for target_quadrant in range(4):
            rewards_quadrant = rewards[quadrants==target_quadrant]
            if len(rewards_quadrant) != 0:
                mean_rewards_quadrant = torch.mean(torch.FloatTensor(rewards_quadrant))
                
                if mean_rewards_quadrant >= Running_mean[target_quadrant]:
                    Running_mean[target_quadrant] = decay * Running_mean[target_quadrant] + (1-decay) * mean_rewards_quadrant
                
                else:
                    decay = 1 - (1-decay)/10
                    Running_mean[target_quadrant] = decay * Running_mean[target_quadrant] + (1-decay) * mean_rewards_quadrant
                # assert self.Running_mean[target_quadrant]
                
        self.register_buffer("Running_mean", Running_mean)
        
    def normalize(self, rewards, quadrants):
        Running_mean = self.get_buffer("Running_mean")
        scaled_reward = dc(rewards)
        for target_quadrant in range(4):
            cond = (quadrants == target_quadrant)
            scaled_reward[cond] = scaled_reward[cond] - Running_mean[target_quadrant]
            # scaled_reward[cond] = scaled_reward[cond]/ (torch.maximum(torch.std(scaled_reward[cond]), torch.tensor(1e-6)))
            # scaled_reward[cond] = torch.nn.LeakyReLU()(scaled_reward[cond])
            scaled_reward[cond] = torch.nn.ReLU()(scaled_reward[cond])
            
        return scaled_reward

class WhiteningNormalizer():
    def normalize(self, reward):
        scaled_rewards = (reward-torch.mean(reward))/(torch.std(reward)+1e-6)
        scaled_rewards = torch.nn.Softplus()(scaled_rewards)