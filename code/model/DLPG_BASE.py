# %%
import copy
from typing import Union

import torch
from torch import nn
from torch import Tensor
from torch.distributions import Normal, Laplace, MultivariateNormal
from utils.tools import get_linear_layer


from configs.template import DLPG_ARGS_Template, DLPG_MDN_ARGS_Template, SAC_ARGS_Template
from typing import Union

import copy


import torch
import copy
from torch.nn import Module

### Roll out ###
from typing import Union
from envs.reacherEnv import CustomReacherEnv
import numpy as np



class DLPG_ABC_(nn.Module):
    def __init__(self, args:Union[DLPG_ARGS_Template, DLPG_MDN_ARGS_Template]):
        super(DLPG_ABC_,self).__init__()        
        self.args = args
        # layers
        self.C_Embed               = nn.Embedding(4,args.cdim).to(args.device)
        self.C_Embed.weight.data   = torch.eye(args.cdim)[:4]
        self.C_Embed.weight.requires_grad_(False)
        self.set_Encoder(args)        
        self.set_Decoder(args)
        self.set_Optimizer(args)
        
    def set_Encoder(self, args:Union[DLPG_ARGS_Template, DLPG_MDN_ARGS_Template,]):
        self.encoder           = Encoder(
            args.anchor_dim,
            args.cdim,
            args.zdim,
            args.hdim
            ).to(args.device)
        self.prior_distribution= torch.distributions.Normal(
            loc = torch.zeros(1,args.zdim).to(args.device),
            scale = torch.ones(1,args.zdim).to(args.device))

    def set_Decoder(self, args:Union[DLPG_ARGS_Template, DLPG_MDN_ARGS_Template,]):
        raise NotImplementedError()
    
    def set_Optimizer(self,args):
        self.optimizer = torch.optim.Adam(self.parameters(),lr= args.lr)
        
    def forward(self, anchor:Tensor, quadrant:Tensor):
        raise NotImplementedError()
    
    def encode(self, anchor:Tensor, quadrant:Tensor):
        raise NotImplementedError()
    
    def decode(self, z_distribution, c_embedding):
        raise NotImplementedError()
    
    @torch.no_grad()
    def exploit(self,quadrant=0, sample_number=1):
        """
        input[0]: target quadrant to roll out
        input[1]: number of samples to roll out
        
        return[0]: {sample_number} of anchors
        return[1]: failure ratio of exploiting out of joint limit
        """
        raise NotImplementedError()
    
    def get_randomachors(self):
        jointlimit = torch.FloatTensor(self.args.jointlimit)
        random_anchor = (torch.rand(2).to(self.args.device)-0.5)*2* jointlimit
        return random_anchor.float().unsqueeze(0)
    
    @torch.no_grad()
    def explore(self):
        return self.get_randomachors() 
    
    def save(self, iteration):
        torch.save(self.state_dict(),str(self.args.SAVE_WEIGHT_PATH/f"{iteration}.pth"))

    def save_args(self):
        import yaml
        with open((self.args.SAVE_RESULT_PATH/"args.yaml"), 'w') as f:
            yaml.dump(dict(ARGS = self.args.__dict__), f)
            


    def rollout(self, env:CustomReacherEnv, target_quadrant:int, EXPLOIT:bool, lentraj, RENDER_ROLLOUT):
        if EXPLOIT: 
            anchor,rejection_rate = self.exploit(target_quadrant)
        else:
            anchor = self.explore(); rejection_rate = 0.0
            
        # Get joint_trajectory
        goal_joint_trajectory                = np.linspace(np.zeros_like(anchor),anchor,num=lentraj)
        
        # environment
        env.reset_model(target_quadrant)
        reward,last_quadrant,last_position,joint_trajectory \
            = env.step_trajectory(goal_joint_trajectory=goal_joint_trajectory,RENDER=RENDER_ROLLOUT)

        return (anchor, rejection_rate), (reward,last_quadrant,last_position,joint_trajectory)




# %%
class Encoder(nn.Module):
    def __init__(self,anchor_dim=2,cdim=4,zdim=10, hdim=[128]):
        super(Encoder,self).__init__()
        
        hidden_actv=nn.Tanh; loc_actv=nn.Tanh; scale_actv=nn.Sigmoid
        
        hdim = list(hdim)
        hdim.insert(0,anchor_dim+cdim)
        layers = get_linear_layer(hdim, hidden_actv)
        self.layers = nn.Sequential(*layers)

        self.z_loc       = nn.Sequential(*get_linear_layer(hdim=[hdim[-1], zdim], hidden_actv=loc_actv))
        self.z_log_scale = nn.Sequential(*get_linear_layer(hdim=[hdim[-1], zdim], hidden_actv=scale_actv))
        
    def forward(self,anchor=torch.zeros(128,2),c_embedding=torch.zeros(128,4)) -> Normal:
        anchor_c_concat = torch.cat((anchor,c_embedding),axis=1)
        out = self.layers(anchor_c_concat)
        
        z_loc,z_scale= self.z_loc(out), self.z_log_scale(out)+1e-3
        z_distribution = Normal(z_loc,(z_scale))
        return z_distribution
