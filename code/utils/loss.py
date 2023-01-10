from model.Normalizer import RunningNormalizer, WhiteningNormalizer
import torch
from typing import Union, Dict
from model.DLPG import DLPG
from model.DLPG_MDN import DLPG_MDN
from torch.distributions import kl_divergence

def DLPG_Loss(model:DLPG, batch, normalizer:Union[RunningNormalizer, WhiteningNormalizer]) -> Dict:
    anchor = torch.FloatTensor(batch['anchor'])
    target_quadrant = torch.LongTensor(batch['target_quadrant'])
    reward = torch.FloatTensor( batch['reward'])
    
    # Normalize
    if isinstance(normalizer, RunningNormalizer):
        scaled_rewards = normalizer.normalize(reward, target_quadrant)
    elif isinstance(normalizer, WhiteningNormalizer):
        scaled_rewards = normalizer.normalize(reward)
    else:
        raise NameError(f"normalizer should be instance of either [RunningNormalizer, WhiteningNormalizer]. \n but Found {normalizer.__class__}")
    
    # Get Loss function
    z_distribution, recon_anchor_distribution = model.forward(anchor,target_quadrant)
    unscaled_kld_loss   = torch.sum(kl_divergence(z_distribution, model.prior_distribution), dim=1)
    unscaled_recon_loss = torch.sum(-recon_anchor_distribution.log_prob(anchor), dim=1)
    
    kld_loss            = torch.mean(scaled_rewards * unscaled_kld_loss,  dim=0)
    recon_loss          = torch.mean(scaled_rewards * unscaled_recon_loss,dim=0)
    
    total_loss          = kld_loss + recon_loss
    mean_unscaled_kld_loss   = torch.mean(unscaled_kld_loss,  dim=0)
    mean_unscaled_recon_loss = torch.mean(unscaled_recon_loss,dim=0)
    
    reward_avg          = torch.mean(reward, dim=0)
    
    
    
    # log_dictionary
    log_dictionary = dict(
        unscaled_kld_loss=mean_unscaled_kld_loss,
        unscaled_recon_loss=mean_unscaled_recon_loss,
        kld_loss=kld_loss, 
        recon_loss=recon_loss, 
        total_loss =total_loss,
        reward_avg = reward_avg
    )
    return log_dictionary


def DLPG_MDN_Loss(model:DLPG_MDN, batch, normalizer:Union[RunningNormalizer, WhiteningNormalizer]) -> Dict:
    anchor = torch.FloatTensor(batch['anchor'])
    target_quadrant = torch.LongTensor(batch['target_quadrant'])
    reward = torch.FloatTensor( batch['reward'])
    
    # Normalize
    if isinstance(normalizer, RunningNormalizer):
        scaled_rewards = normalizer.normalize(reward, target_quadrant)
    elif isinstance(normalizer, WhiteningNormalizer):
        scaled_rewards = normalizer.normalize(reward)
    else:
        raise NameError(f"normalizer should be instance of either [RunningNormalizer, WhiteningNormalizer]. \n but Found {normalizer.__class__}")
    
    # Get Loss function
    z_distribution, Mixture_Mode, Mixture_components = model.forward(anchor,target_quadrant)
    unscaled_kld_loss   = torch.sum(kl_divergence(z_distribution, model.prior_distribution), dim=1)
    loglik = Mixture_components.log_prob(anchor.unsqueeze(1).expand_as(Mixture_components.loc))
    loglik = torch.sum(loglik, dim=2)
    unscaled_recon_loss = -torch.logsumexp(Mixture_Mode.logits + loglik, dim=1)
    
    kld_loss            = torch.mean(scaled_rewards * unscaled_kld_loss,  dim=0)
    recon_loss          = torch.mean(scaled_rewards * unscaled_recon_loss,dim=0)
    
    total_loss          = kld_loss + recon_loss
    mean_unscaled_kld_loss   = torch.mean(unscaled_kld_loss,  dim=0)
    mean_unscaled_recon_loss = torch.mean(unscaled_recon_loss,dim=0)
    
    reward_avg          = torch.mean(reward, dim=0)
    
    
    
    # log_dictionary
    log_dictionary = dict(
        unscaled_kld_loss=mean_unscaled_kld_loss,
        unscaled_recon_loss=mean_unscaled_recon_loss,
        kld_loss=kld_loss, 
        recon_loss=recon_loss, 
        total_loss =total_loss,
        reward_avg = reward_avg
    )
    return log_dictionary


