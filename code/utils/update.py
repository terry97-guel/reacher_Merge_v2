from model.Normalizer import RunningNormalizer, WhiteningNormalizer
import torch
from typing import Union, Dict
from model.DLPG import DLPG
from model.DLPG_MDN import DLPG_MDN
from model.SAC import SAC, get_target

from torch.distributions import kl_divergence
from utils.tools import cast_dict_numpy

def DLPG_update(model:DLPG, batch, normalizer:Union[RunningNormalizer, WhiteningNormalizer], TRAIN) -> Dict:
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
    
    # Update
    if TRAIN:
        model.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        model.optimizer.step()
    
    # log_dictionary
    log_dictionary = dict(
        unscaled_kld_loss=mean_unscaled_kld_loss,
        unscaled_recon_loss=mean_unscaled_recon_loss,
        kld_loss=kld_loss, 
        recon_loss=recon_loss, 
        total_loss =total_loss,
        reward_avg = reward_avg
    )
    return cast_dict_numpy(log_dictionary)


def DLPG_MDN_update(model:DLPG_MDN, batch, normalizer:Union[RunningNormalizer, WhiteningNormalizer], TRAIN) -> Dict:
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
    
    # Update
    if TRAIN:
        model.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        model.optimizer.step()
    
    # log_dictionary
    log_dictionary = dict(
        unscaled_kld_loss=mean_unscaled_kld_loss,
        unscaled_recon_loss=mean_unscaled_recon_loss,
        kld_loss=kld_loss, 
        recon_loss=recon_loss, 
        total_loss =total_loss,
        reward_avg = reward_avg
    )
    return cast_dict_numpy(log_dictionary)


def SAC_update(model: SAC, batch, TRAIN:bool):
    td_target = get_target(model.pi, model.q1_target, model.q2_target, batch)
    
    critic_loss_1 = model.q1.update(td_target, batch, TRAIN)
    critic_loss_2 = model.q2.update(td_target, batch, TRAIN)
    
    actor_loss, alpha_loss = model.pi.update(model.q1, model.q2,batch, TRAIN)
    
    if TRAIN:
        model.q1.soft_update(model.q1_target)
        model.q2.soft_update(model.q2_target)
    
    
    reward = torch.FloatTensor(batch['reward'])
    reward_avg          = torch.mean(reward, dim=0)
    
    log_dictionary = dict(
        actor_loss = actor_loss,
        alpha_loss = alpha_loss,
        critic_loss_1 = critic_loss_1,
        critic_loss_2 = critic_loss_2,
        reward_avg = reward_avg
    )
    
    return cast_dict_numpy(log_dictionary)


def PPO_update(model, batch, TRAIN=False):
    
    
    
    
    log_dictionary = dict(
        
    )
    
    return cast_dict_numpy