import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from utils.tools import cast_numpy
from configs.template import PPO_ARGS_Template
from envs.reacherEnv import get_quadrant, CustomReacherEnv


class PPO(nn.Module):
    def __init__(self, args:PPO_ARGS_Template):
        super().__init__()
        
        self.args = args
        self.clip_ratio = args.clip_ratio
        
        
        self.pi = PPO_Actor(args=args).to(args.device)
        self.q  = CriticClass(args=args).to(args.device)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        
    def rollout(self,env:CustomReacherEnv, target_quadrant:int, lentraj, RENDER_ROLLOUT):
        anchor,log_p,rejection_rate = self.pi.get_anchor(target_quadrant)
        
        value = self.q(torch.LongTensor([target_quadrant]).to(self.args.device))
        # Get joint_trajectory
        goal_joint_trajectory                = np.linspace(np.zeros_like(anchor),anchor,num=lentraj)
        
        # environment
        env.reset_model(target_quadrant)
        reward,last_quadrant,last_position,joint_trajectory \
            = env.step_trajectory(goal_joint_trajectory=goal_joint_trajectory,RENDER=RENDER_ROLLOUT)

        return (anchor, log_p, value, rejection_rate), (reward,last_quadrant,last_position,joint_trajectory)
    
    def save(self, iteration):
        torch.save(self.state_dict(),str(self.args.SAVE_WEIGHT_PATH/f"{iteration}.pth"))
    
    def forward(self, target_quadrant):
        anchor_distribution = self.pi(target_quadrant)
        value  = self.q(target_quadrant)
        
        return anchor_distribution, value
        
    def update(self, batch ,TRAIN):
        reward = torch.FloatTensor(batch['reward']).to(self.args.device)
        value_old = torch.FloatTensor(batch['value']).to(self.args.device)
        target_quadrant = torch.LongTensor(batch['target_quadrant']).to(self.args.device)
        anchor = torch.FloatTensor(batch['anchor']).to(self.args.device)
        logp_old = torch.FloatTensor(batch["log_p"]).to(self.args.device)
        
        adv = (reward - value_old)[..., None]
        anchor_distribution,value = self.forward(target_quadrant)
        
        logp_a_curr = anchor_distribution.log_prob(anchor)
        entropy = anchor_distribution.entropy().mean()
        
        ratio  = (logp_a_curr-logp_old).exp()
        
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0-self.clip_ratio, 1.0+self.clip_ratio)*adv
        # Actor 
        actor_loss = -torch.minimum(surr1,surr2).mean()
        
        # Critic
        critic_loss = (reward - value).pow(2).mean()
        total_loss = 0.5*critic_loss+actor_loss-0.001*entropy
        
        if TRAIN:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        return actor_loss, critic_loss, total_loss
        
        
class CriticClass(nn.Module):
    def __init__(self, args:PPO_ARGS_Template):
        super(CriticClass, self).__init__()
        self.args = args

        self.fc_s = nn.Linear(args.cdim,128)
        self.fc_out = nn.Linear(128,1)

        self.C_Embed               = nn.Embedding(4,args.cdim).to(args.device)
        self.C_Embed.weight.data   = torch.eye(args.cdim)[:4]
        self.C_Embed.weight.requires_grad_(False)

    def forward(self, target_quadrant):
        c_emb= self.C_Embed(target_quadrant)
        c_emb = F.relu(self.fc_s(c_emb))
        s_value = self.fc_out(c_emb)
        
        return s_value 
    
class PPO_Actor(nn.Module):
    def __init__(self, args:PPO_ARGS_Template):
        super(PPO_Actor, self).__init__()
        self.args = args
        
        self.fc1 = nn.Linear(args.cdim,256)
        self.fc2 = nn.Linear(256,256)
        self.fc_mean = nn.Linear(256,args.anchor_dim)
        self.fc_std = nn.Linear(256,args.anchor_dim)

        self.C_Embed               = nn.Embedding(4,args.cdim).to(args.device)
        self.C_Embed.weight.data   = torch.eye(args.cdim)[:4]
        self.C_Embed.weight.requires_grad_(False)
    
    def forward(self,target_quadrant):
        c_emb = self.C_Embed(target_quadrant)
        c_emb = F.relu(self.fc1(c_emb))
        c_emb = F.relu(self.fc2(c_emb))
        anchor_loc = F.tanh(self.fc_mean(c_emb)) * self.args.jointlimit
        # std : softplus or ReLU activate function [Because It should be higher than 0]
        anchor_scale = F.softplus(self.fc_std(c_emb))
        anchor_distribution = Normal(anchor_loc,anchor_scale)

        return anchor_distribution
    
    @torch.no_grad()
    def get_anchor(self,target_quadrant:int):
        target_quadrant = torch.LongTensor([target_quadrant]).to(self.args.device)
        trial_      = 0
        reject_num_ = 0
        for _ in range(100):
            trial_ = trial_ + 1
            
            anchor_distribution = self.forward(target_quadrant)
            anchor = anchor_distribution.rsample()
            log_p = anchor_distribution.log_prob(anchor)
            
            # Sample Until it satisfy jointlimit
            if (abs(self.args.jointlimit) - abs(anchor)>0).all(): 
                return anchor.cpu().numpy(), log_p.cpu().numpy(), reject_num_/trial_
            
            else: 
                # print("Reject Sample!!")
                reject_num_ = reject_num_+1
    
