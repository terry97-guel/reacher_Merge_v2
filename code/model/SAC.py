import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from utils.tools import cast_numpy
from configs.template import SAC_ARGS_Template
from envs.reacherEnv import get_quadrant, CustomReacherEnv


class SAC(nn.Module):
    def __init__(self, args:SAC_ARGS_Template):
        super(SAC,self).__init__()
        self.args = args
        self.q1, self.q1_target = CriticClass(args).to(args.device), CriticClass(args).to(args.device)
        self.q2, self.q2_target =  CriticClass(args).to(args.device), CriticClass(args).to(args.device)
        self.pi = ActorClass(args=args).to(args.device)   
         
    def rollout(self,env:CustomReacherEnv, target_quadrant:int, lentraj, RENDER_ROLLOUT):
        anchor,rejection_rate = self.pi.get_anchor(target_quadrant)
        
        # Get joint_trajectory
        goal_joint_trajectory                = np.linspace(np.zeros_like(anchor),anchor,num=lentraj)
        
        # environment
        env.reset_model(target_quadrant)
        reward,last_quadrant,last_position,joint_trajectory \
            = env.step_trajectory(goal_joint_trajectory=goal_joint_trajectory,RENDER=RENDER_ROLLOUT)

        return (anchor, rejection_rate), (reward,last_quadrant,last_position,joint_trajectory)
    
    def save(self, iteration):
        torch.save(self.state_dict(),str(self.args.SAVE_WEIGHT_PATH/f"{iteration}.pth"))
        
    

class ActorClass(nn.Module):
    def __init__(self, args: SAC_ARGS_Template):
        super(ActorClass,self).__init__()
        # Gaussian Distribution
        self.args = args
        
        self.fc1 = nn.Linear(args.cdim,256)
        self.fc2 = nn.Linear(256,256)
        self.fc_mean = nn.Linear(256,args.anchor_dim)
        self.fc_std = nn.Linear(256,args.anchor_dim)
        self.optimizer = optim.Adam(self.parameters(),lr=args.lr)
        self.target_entropy = - args.anchor_dim

        # Autotuning Alpha
        self.log_alpha = torch.tensor(np.log(args.init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha],lr = args.lr_alpha)  
        
        self.C_Embed               = nn.Embedding(4,args.cdim).to(args.device)
        self.C_Embed.weight.data   = torch.eye(args.cdim)[:4]
        self.C_Embed.weight.requires_grad_(False)

    def forward(self,target_quadrant):
        c_emb = self.C_Embed(target_quadrant)
        c_emb = F.relu(self.fc1(c_emb))
        c_emb = F.relu(self.fc2(c_emb))
        anchor_loc = self.fc_mean(c_emb)
        # std : softplus or ReLU activate function [Because It should be higher than 0]
        anchor_scale = F.softplus(self.fc_std(c_emb))
        anchor_distribution = Normal(anchor_loc,anchor_scale)
        anchor = anchor_distribution.rsample()
        log_prob = anchor_distribution.log_prob(anchor)
        
        # Change of variable
        real_anchor = torch.tanh(anchor) * self.args.jointlimit
        real_log_prob = log_prob - torch.log(1-torch.tanh(anchor).pow(2) + 1e-7)
        return real_anchor, real_log_prob

    def update(self, q1, q2, batch, TRAIN:bool):
        s = torch.LongTensor(batch['target_quadrant']).to(self.args.device)
        a = torch.FloatTensor(batch['anchor']).to(self.args.device)

        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s,a), q2(s,a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = (-min_q - entropy).mean()
        
        if TRAIN:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        
        if TRAIN:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
        
        return cast_numpy(loss), cast_numpy(alpha_loss)
    
    @torch.no_grad()
    def get_anchor(self,target_quadrant:int):
        target_quadrant = torch.LongTensor([target_quadrant]).to(self.args.device)
        trial_      = 0
        reject_num_ = 0
        while True:
            trial_ = trial_ + 1
            anchor, _ = self.forward(target_quadrant)

            # Sample Until it satisfy jointlimit
            if (abs(self.args.jointlimit) - abs(anchor)>0).all(): 
                return anchor.cpu().numpy(), reject_num_/trial_
            
            else: 
                # print("Reject Sample!!")
                reject_num_ = reject_num_+1


class CriticClass(nn.Module):
    def __init__(self, args:SAC_ARGS_Template):
        super(CriticClass,self).__init__()
        self.args = args

        self.fc_s = nn.Linear(args.cdim,128)
        self.fc_a = nn.Linear(args.anchor_dim,128)
        self.fc_cat = nn.Linear(256,256)
        self.fc_out = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(),lr=args.lr)
        self.tau = args.tau

        self.C_Embed               = nn.Embedding(4,args.cdim).to(args.device)
        self.C_Embed.weight.data   = torch.eye(args.cdim)[:4]
        self.C_Embed.weight.requires_grad_(False)

    def forward(self,target_quadrant,anchor):
        c_emb= self.C_Embed(target_quadrant)

        c_emb = F.relu(self.fc_s(c_emb))
        anchor_emb= F.relu(self.fc_a(anchor))
        ac_emb = torch.cat([c_emb,anchor_emb], dim=1)
        q = F.relu(self.fc_cat(ac_emb))
        q_value = self.fc_out(q)

        return q_value

    def update(self,target, batch, TRAIN:bool):
        s = torch.LongTensor(batch['target_quadrant']).to(self.args.device)
        a = torch.FloatTensor(batch['anchor']).to(self.args.device)

        loss = F.smooth_l1_loss(self.forward(s,a).flatten(), target).mean()
        if TRAIN:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss

    # DDPG soft_update
    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

def get_target(pi, q1, q2, batch, gamma=0.99):
    r = torch.FloatTensor(batch["reward"])
    s = torch.LongTensor(batch['target_quadrant'])
    
    
    current_quadrant = get_quadrant(batch['last_position'].flatten())
    done = current_quadrant != s

    s_prime = s


    with torch.no_grad():
        a_prime, log_prob= pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s_prime,a_prime), q2(s_prime,a_prime)
        q = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q, 1)[0]
        target = r + gamma * done * (min_q + torch.mean(entropy, dim=1))
    return target 