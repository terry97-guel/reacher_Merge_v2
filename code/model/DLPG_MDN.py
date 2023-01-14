# %%
from random import sample
import torch
from torch import nn
import numpy as np
from utils.tools import cast_tensor, get_linear_layer
import copy
from typing import Tuple, Union
import torch
import torch.nn as nn
from torch.distributions import OneHotCategorical, Normal, MultivariateNormal, Laplace

from configs.template import  DLPG_MDN_ARGS_Template
from typing import Union

from model.DLPG_BASE import DLPG_ABC_


class DLPG_MDN(DLPG_ABC_):
    def __init__(self, args:DLPG_MDN_ARGS_Template):
        super(DLPG_MDN,self).__init__(args)

    def set_Decoder(self, args: DLPG_MDN_ARGS_Template):
        self.decoder           = MDN_Decoder(
            args.anchor_dim,
            args.cdim,
            args.zdim,
            args.hdim,
            args.jointlimit,
            args.n_components).to(self.args.device)
        
    
    def forward(
        self,anchor=torch.zeros(128,2), quadrant=np.zeros(128)
        ) -> Tuple[Normal, OneHotCategorical, Normal ]:
        z_distribution, c_embedding      = self.encode(anchor, quadrant)
        Mixture_Mode, Mixture_components = self.decode(z_distribution, c_embedding)
        
        return z_distribution, Mixture_Mode, Mixture_components

    def encode(self,anchor=torch.zeros(128,2), quadrant=np.zeros(128)):
        quadrant                   = cast_tensor(quadrant).long().to(self.args.device)
        c_embedding                = self.C_Embed(quadrant)
        z_distribution             = self.encoder(anchor,c_embedding)

        return z_distribution, c_embedding
    
    def decode(
        self, 
        z_distribution,
        c_embedding
        ) -> Tuple[OneHotCategorical, Normal]:
        '''
        Decode: (z_sample, c_embedding) -> MDN distribution of anchor 
        
        z_sample: sampled value from z_distribution
        c_embedding: one-hot vector of quadrant
        Mixture_Mode: Categorical distribution of MDN Modes.
        Mixture_components: Normal distribution of each MDN components.

        '''
        z_sample = z_distribution.rsample()
        
        Mixture_Mode, Mixture_components  = self.decoder(z_sample,c_embedding)

        return Mixture_Mode, Mixture_components
    
    @torch.no_grad()
    def exploit(self,quadrant=0):
        quadrant                         = torch.tensor([quadrant]).long().to(self.args.device)
        c_embedding                      = self.C_Embed(quadrant)
        z_sample                         = self.prior_distribution.sample().to(self.args.device)
        
        trial_      = 0
        reject_num_ = 0
        while True:
            trial_ = trial_ + 1
        
            recon_anchor    = self.decoder.sample(z_sample,c_embedding)
            
            # Sample Until it satisfy jointlimit
            if (abs(self.args.jointlimit) - abs(recon_anchor)>0).all(): 
                return recon_anchor.cpu().numpy(), reject_num_/trial_
            
            else: 
                # print("Reject Sample!!")
                reject_num_ = reject_num_+1
    

# %%
class MDN_Decoder(nn.Module):
    """
    Mixture density network.

    [ Bishop, 1994 ]

    n_components: int; number of components in the mixture model
    """
    def __init__(self,anchor_dim=2,cdim=4,zdim=10, hdim=[128],jointlimit=torch.tensor([3.14,2.22]),n_components = 5):
        super().__init__()
        self.pi_network = CategoricalNetwork(
            anchor_dim=anchor_dim, cdim=cdim,zdim=zdim, hdim=hdim, n_components = n_components)
        
        self.normal_network = MixtureComponentNetwork(
            anchor_dim=anchor_dim, cdim=cdim,zdim=zdim, hdim=hdim,jointlimit = jointlimit,n_components = n_components)

    def forward(self, z_sample,c_embedding) -> Tuple[OneHotCategorical, Union[Normal,Laplace]]:
        return self.pi_network(z_sample,c_embedding), self.normal_network(z_sample,c_embedding)


    def sample(self, z_sample, c_embedding):
        '''
        When sampling, we use mode of MDN components.
        Instead of sampling from MDN components 
        '''
        pi, normal = self.forward(z_sample, c_embedding)
        samples = torch.sum(pi.sample().unsqueeze(2) * normal.loc, dim=1)

        return samples

class MixtureComponentNetwork(nn.Module):
    def __init__(self,anchor_dim=2,cdim=4,zdim=10, hdim=[128],jointlimit=torch.tensor([3.14,2.22]),n_components = 5):
        super().__init__()
        self.jointlimit   = jointlimit
        self.n_components = n_components
        self.anchor_dim   = anchor_dim
        hdim = list(hdim)
        hdim.append(zdim+cdim)
        
        std = 0.1
        self.layers          = nn.Sequential(*get_linear_layer(list(reversed(hdim)), nn.Tanh, std=std))
        
        self.recon_loc       = nn.Sequential(*get_linear_layer(hdim=[hdim[0], n_components * anchor_dim], hidden_actv=nn.Tanh,  std=std))
        self.recon_log_scale = nn.Sequential(*get_linear_layer(hdim=[hdim[0], n_components * anchor_dim], hidden_actv=nn.Sigmoid, std=std))
        

    def forward(self, z_sample,c_embedding) -> Union[Normal, Laplace]:
        '''
        Latent variable z and conditional vector c are fed to decoder,
        resulting anchor distribution of anchor_loc(mean), anchor_scale(std).

        anchor_loc is scaled according to joint limit.
        anchor_scale is scaled in similar to SIGMA VAE(https://arxiv.org/pdf/2006.13202.pdf)

        To leverge auto-calibration of Beta-VAE, SIGMA VAE is used for every MDN components. 
        Possible betas range (0.5~2)
        
        '''
        # Feed
        z_c_concat = torch.cat((z_sample,c_embedding),axis=1)
        out = self.layers(z_c_concat)

        anchor_loc   = self.recon_loc(out).reshape(-1, self.n_components, self.anchor_dim)
        anchor_scale = self.recon_log_scale(out).reshape(-1, self.n_components, self.anchor_dim)
        
        # Scale
        anchor_loc  = anchor_loc*self.jointlimit
        
        betas = [0.5, 2]
        lower_bound = torch.sqrt(torch.tensor(betas[0])); upper_bound = torch.sqrt(torch.tensor(betas[1]))
        anchor_scale = anchor_scale * (upper_bound-lower_bound)/2 + (upper_bound+lower_bound)/2 

        # Get anchor distribution
        recon_anchor_distribution = Normal(anchor_loc,anchor_scale)

        return recon_anchor_distribution

class CategoricalNetwork(nn.Module):
    def __init__(self,anchor_dim=2,cdim=4,zdim=10, hdim=[128], n_components = 5):
        super().__init__()
        self.n_components = n_components
        self.anchor_dim   = anchor_dim
        hdim = list(hdim)
        hdim.append(zdim+cdim)
        
        layers            = get_linear_layer(list(reversed(hdim)), hidden_actv=nn.Tanh)
        self.layers       = nn.Sequential(*layers)
        
        self.MixtureLogit = nn.Sequential(*get_linear_layer(hdim=[hdim[0], n_components], hidden_actv=nn.Softmax))
        
    def forward(self, z_sample,c_embedding) -> OneHotCategorical:
        z_c_concat = torch.cat((z_sample,c_embedding),axis=1)
        out = self.layers(z_c_concat)
        logits = self.MixtureLogit(out)
        return OneHotCategorical(logits=logits)



#%%
if __name__=="__main__":
    pass