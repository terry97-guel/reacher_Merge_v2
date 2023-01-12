# %%
from random import sample
import torch
from torch import nn
import numpy as np
from utils.tools import cast_tensor, get_runname, get_linear_layer
import copy


from model.DLPG_BASE import DLPG_ABC_
from configs.template import DLPG_ARGS_Template
from model.Normalizer import RunningNormalizer, WhiteningNormalizer


class DLPG(DLPG_ABC_):
    def __init__(self,args:DLPG_ARGS_Template):
        super().__init__(args)
    
    
    def set_Decoder(self, args:DLPG_ARGS_Template):
        self.decoder                = Decoder(
            args.anchor_dim,
            args.cdim,
            args.zdim,
            args.hdim,
            args.jointlimit, 
            ).to(self.args.device)
        
    def forward(self,anchor=torch.zeros(128,2), quadrant=torch.zeros(128)):
        z_distribution, c_embedding = self.encode(anchor, quadrant)
        recon_anchor_distribution   = self.decode(z_distribution, c_embedding)
        
        return z_distribution, recon_anchor_distribution

    def encode(self,anchor=torch.zeros(128,2), quadrant=torch.zeros(128)):
        c_embedding                = self.C_Embed(quadrant)
        z_distribution             = self.encoder(anchor,c_embedding)

        return z_distribution, c_embedding
    
    def decode(self, z_distribution, c_embedding):
        z_sample                   = z_distribution.rsample()
        recon_anchor_distribution  = self.decoder(z_sample,c_embedding)
        
        return recon_anchor_distribution
    
    
    @torch.no_grad()
    def exploit(self,quadrant=0):
        quadrant                         = torch.tensor([quadrant]).long().to(self.args.device)
        c_embedding                      = self.C_Embed(quadrant)
        z_sample                         = self.prior_distribution.sample().to(self.args.device)
        
        trial_      = 0
        reject_num_ = 0
        while True:
            trial_ = trial_ + 1
            recon_anchor_distribution    = self.decoder(z_sample,c_embedding)
            
            # Sample from recon_anchor_distribution
            recon_anchor = recon_anchor_distribution.loc  
            
            # Sample Until it satisfy jointlimit
            jointlimit = torch.FloatTensor(self.args.jointlimit)
            if (abs(jointlimit) - abs(recon_anchor)>0).all(): 
                return recon_anchor.cpu().numpy(), reject_num_/trial_
            
            else: 
                # print("Reject Sample!!")
                reject_num_ = reject_num_+1



# %%
from typing import Union
from torch.distributions import Normal, Laplace


class Decoder(nn.Module):
    def __init__(self,anchor_dim=2,cdim=4,zdim=10, hdim=[128],jointlimit=torch.tensor([3.14,2.22]),
                 hidden_actv=nn.Tanh, loc_actv=nn.Tanh):
        super(Decoder,self).__init__()
        self.jointlimit = jointlimit
        
        std = 0.3
        
        hdim = list(hdim)
        hdim.append(zdim+cdim)
        layers = get_linear_layer(list(reversed(hdim)), hidden_actv, std=std)
        self.layers = nn.Sequential(*layers)
        
        self.recon_loc       = nn.Sequential(*get_linear_layer(hdim=[hdim[0], anchor_dim], hidden_actv=loc_actv,  std=std))
        
    def forward(self,z_sample=torch.zeros(128,10),c_embedding=torch.zeros(128,4)) -> Union[Normal, Laplace]:
        z_c_concat = torch.cat((z_sample,c_embedding),axis=1)
        out = self.layers(z_c_concat)
        
        # Parameter of Laplacian distribution (https://en.wikipedia.org/wiki/Laplace_distribution#Moments)
        anchor_loc = self.recon_loc(out)*self.jointlimit
        anchor_scale = torch.ones_like(anchor_loc, dtype=torch.float)
        recon_anchor_distribution = Normal(anchor_loc,anchor_scale)
        return recon_anchor_distribution
