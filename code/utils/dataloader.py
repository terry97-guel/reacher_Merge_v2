import numpy as np
import torch
import sys
from pathlib import Path

from scipy.spatial import distance 
import random as rd 
import torch 
import copy

from typing import List
from numpy import ndarray

class Buffer: 
    def __init__(self, anchor_dim=4, last_position_dim=2, buffer_limit=1000):
        self.anchor_dim = anchor_dim
        self.last_position_dim = last_position_dim
        self.buffer_limit = buffer_limit
        
        
        self.anchor                              =   np.zeros([buffer_limit, anchor_dim], dtype=np.float32)
        self.target_quadrant                     =   np.zeros([buffer_limit], dtype=np.float32)
        self.reward                              =   np.zeros([buffer_limit], dtype=np.float32)
        self.last_position                       =   np.zeros([buffer_limit, last_position_dim], dtype=np.float32)
        self.ptr, self.size, self.buffer_limit   =   0, 0, buffer_limit 

    def store(self, anchor, reward, target_quadrant, last_position):
        self.anchor[self.ptr]                = anchor
        self.target_quadrant[self.ptr]       = target_quadrant
        self.reward[self.ptr]                = reward
        self.last_position[self.ptr]         = last_position
        self.ptr                             = (self.ptr+1)%self.buffer_limit      # pointer of last saved idx
        self.size                            = min(self.size+1, self.buffer_limit) # number of instance stored
        
    def __len__(self):
        return self.size

    def empty(self):
        anchor_dim = self.anchor_dim
        last_position_dim = self.last_position_dim
        buffer_limit = self.buffer_limit
        
        self.anchor                              =   np.zeros([buffer_limit, anchor_dim], dtype=np.float32)
        self.target_quadrant                     =   np.zeros([buffer_limit], dtype=np.float32)
        self.reward                              =   np.zeros([buffer_limit], dtype=np.float32)
        self.last_position                       =   np.zeros([buffer_limit, last_position_dim], dtype=np.float32)
        self.ptr, self.size, self.buffer_limit   =   0, 0, buffer_limit 

class Sampler:
    @staticmethod
    def sample_from_idx(buffer:Buffer, idxs):
        batch = dict(
            anchor              = buffer.anchor[idxs],
            target_quadrant     = buffer.target_quadrant[idxs],
            reward              = buffer.reward[idxs],
            last_position       = buffer.last_position[idxs],
            )
        
        return batch
        
    @staticmethod
    def sample_all(buffer:Buffer):
        sel_idxs = np.arange(buffer.size)
        return Sampler.sample_from_idx(buffer, sel_idxs)
    
    @staticmethod
    def sample_random(buffer:Buffer, batch_size):
        sel_idxs = np.random.permutation(buffer.size)[:batch_size]
        return Sampler.sample_from_idx(buffer, sel_idxs)
    
    '''
    @staticmethod
    def sample_Frontier_LAdpp(
        train_buffer:Buffer,
        test_buffer:Buffer,
        test_points_ratio=1.0,
        batch_size=128,
        train_number_ratio= 1.0,
        hyp=dict(k_gain=10.0)
        ):
        batch_size = 1
        test_points_number = int(len(test_buffer) * test_points_ratio)
        train_points_number = max(int(len(train_buffer) * train_number_ratio), batch_size)
        
        # Concate partial data from test and  partial train data
        def random_choice_2d(target, number, replace):
            target_idx = np.random.choice(np.arange(len(target)), number, replace=replace)
            return target[target_idx]
        
        LAdpp_target = np.concatenate(
            [random_choice_2d(test_buffer.last_position[:test_buffer.size],   test_points_number,  replace=False),
             random_choice_2d(train_buffer.last_position[:train_buffer.size], train_points_number, replace=False)]) 
        
        assert len(LAdpp_target) == test_points_number + train_points_number
        
        # data from test are already chosen.
        init_idxs = tuple(range(test_points_number)) 
        
        def LAdpp(target:ndarray, init_idxs:tuple, batch_size, hyp=dict(k_gain=10.0), train_buffer=None):
            # Define Kernel
            D      = distance.cdist(target,target,'sqeuclidean')
            K_MAT  = np.exp(-hyp["k_gain"]*D)
            
            n      = target.shape[0]
            sum_K  = np.zeros(n)
            
            # get sum_k
            for idx in init_idxs:
                curr_K = K_MAT[idx,:]
                sum_K = sum_K+curr_K
            
            remain_idx = np.arange(len(init_idxs), len(target))
            
            sel_idxs = []
            for _ in range(batch_size):
                if len(init_idxs) == 0:
                    index_of_remain_idx = np.random.randint(len(remain_idx))
                else:
                    k_val = sum_K[remain_idx]
                    index_of_remain_idx = np.argmin(k_val)
                    
                idx_ = remain_idx[index_of_remain_idx]
                curr_K = K_MAT[idx_,:]
                sum_K = sum_K+curr_K
                
                sel_idxs.append(idx_ - len(init_idxs))
                assert (target[idx_] == target[len(init_idxs):][idx_ - len(init_idxs)]).all()
                remain_idx = np.delete(remain_idx, index_of_remain_idx)
            return tuple(sel_idxs)
        
        sel_idxs = LAdpp(LAdpp_target, init_idxs, batch_size, hyp, train_buffer)
        
        # Must subract idxs of test_buffer because we want idxs of train budffer
        # sel_idxs = np.array(sel_idxs)
        
        # sel_idxs =  sel_idxs[sel_idxs >= test_points_number]
        # assert len(sel_idxs) == batch_size
        
        # sel_idxs = sel_idxs - test_points_number
        return Sampler.sample_from_idx(train_buffer, sel_idxs)
    
    '''
    
    @staticmethod
    def sample_Frontier_LAdpp(
        train_buffer:Buffer,
        test_buffer:Buffer,
        test_points_ratio=1.0,
        batch_size=128,
        train_number_ratio= 1.0,
        hyp=dict(k_gain=10.0)
        ):
        test_points_number = int(len(test_buffer) * test_points_ratio)
        train_points_number = int(len(train_buffer) * train_number_ratio)
        assert train_points_number >= batch_size
        
        # Concate partial data from test and  partial train data
        test_choice  = np.random.choice(np.arange(len(test_buffer)), test_points_number)
        train_choice = np.random.choice(np.arange(len(train_buffer)), train_points_number)
        
        target = np.concatenate(
            [test_buffer.anchor[test_choice], train_buffer.anchor[train_choice]])
        
        assert len(target) == test_points_number + train_points_number
        init_idxs = tuple(range(test_points_number)) 
        
        def LAdpp(target:ndarray, init_idxs:tuple, batch_size, hyp=dict(k_gain=10.0)):
            # Define Kernel
            D      = distance.cdist(target,target,'sqeuclidean')
            K_MAT  = np.exp(-hyp["k_gain"]*D)
            
            n      = target.shape[0]
            sum_K  = np.zeros(n)
            init_number = len(init_idxs)
            # get sum_k from test
            for idx in range(init_number):
                curr_K = K_MAT[idx,:]
                sum_K = sum_K+curr_K
            
            remain_idx = np.arange(init_number, len(target))
            
            sel_idxs = []
            for _ in range(batch_size):
                if init_number == 0:
                    index_of_remain_idx = np.random.randint(len(remain_idx))
                else:
                    k_val = sum_K[remain_idx]
                    index_of_remain_idx = np.argmin(k_val)
                    
                idx_ = remain_idx[index_of_remain_idx]
                curr_K = K_MAT[idx_,:]
                sum_K = sum_K+curr_K
                
                sel_idxs.append(idx_ - init_number)
                remain_idx = np.delete(remain_idx, index_of_remain_idx)
                
            last_idx_for_test = idx_
            return sel_idxs, last_idx_for_test
        
        sel_idxs, last_idx_for_test = LAdpp(target, init_idxs, batch_size, hyp)
        sel_train_idxs = train_choice[sel_idxs]
        
        assert (train_buffer.anchor[sel_train_idxs[-1]] == target[last_idx_for_test]).all()
        return Sampler.sample_from_idx(train_buffer, sel_train_idxs)
    
# plot_samples(test_buffer, train_buffer, np.arange(train_buffer.size), name="all")
# plot_samples(test_buffer, train_buffer, np.random.permutation(train_buffer.size)[:batch_size], name="random")
# plot_samples(test_buffer, train_buffer, sel_train_idxs, name="Fdpp")

# %% Plot code
def plot_samples(test_buffer, train_buffer, sel_idxs, name="temp"):
    from matplotlib import pyplot as plt
    batch = Sampler.sample_from_idx(train_buffer, sel_idxs)
    plt.figure()
    # plt.xlim([-0.22,0.22])
    # plt.ylim([-0.22,0.22])
    last_position = batch['last_position']
    plt.scatter(last_position[:,0],last_position[:,1], color = 'k', marker = 'o')
    
    batch = Sampler.sample_all(test_buffer)
    last_position = batch['last_position']
    plt.scatter(last_position[:,0],last_position[:,1], color = 'r', marker = 'x')
    
    plt.savefig(name)
    
def plot_anchors(test_buffer, train_buffer, sel_idxs, name="temp"):
    from matplotlib import pyplot as plt
    batch = Sampler.sample_from_idx(train_buffer, sel_idxs)
    plt.figure()
    # plt.xlim([-0.22,0.22])
    # plt.ylim([-0.22,0.22])
    last_position = batch['anchor']
    plt.scatter(last_position[:,0],last_position[:,1], color = 'k', marker = 'o')
    
    batch = Sampler.sample_all(test_buffer)
    last_position = batch['anchor']
    plt.scatter(last_position[:,0],last_position[:,1], color = 'r', marker = 'x')
    
    plt.savefig(name)