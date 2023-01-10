# %%
from pathlib import Path
import sys
import argparse
import wandb
from utils.dataloader import Buffer, Sampler
from model.DLPG import DLPG
import os

from envs.reacherEnv import CustomReacherEnv
import numpy as np
import random
import torch

from template.DLPG_Template import DLPG_ARGS_Template
from utils.args import Refine_OptionalArgs_to_Template
from utils.tools import set_seed, set_wandb, print_log_dict, prefix_dict
from utils.path_handler import JUPYTER, RUN, DEBUG, get_BASERDIR
from model.Normalizer import RunningNormalizer, WhiteningNormalizer
from utils.loss import DLPG_Loss
from utils.measure import get_measure


print(sys.version)


def main(args: DLPG_ARGS_Template):
    ### SEED ###
    set_seed(args.random_seed)
    
    ### WANDB ### 
    if args.WANDB:
        set_wandb(pname = args.pname, runname = args.runname)
    
    ### Set Path ###
    if args.SAVE_RESULT_PATH is None:
        args.SAVE_RESULT_PATH    = Path(BASEDIR).parent/"DLPG"/args.runname/"results"    
            
    if args.SAVE_WEIGHT_PATH is None:
        args.SAVE_WEIGHT_PATH = Path(BASEDIR).parent/"DLPG"/args.runname/"weights"

    Path.mkdir(args.SAVE_WEIGHT_PATH,exist_ok=True, parents=True)
    Path.mkdir(args.SAVE_RESULT_PATH,exist_ok=True, parents=True)
    
    
    
    ### Declare Instance ###
    # Buffer
    train_buffer = Buffer(
        anchor_dim=args.anchor_dim, last_position_dim=args.position_dim, buffer_limit=args.buffer_limit, )
    
    test_buffer = Buffer(
        anchor_dim=args.anchor_dim, last_position_dim=args.position_dim, buffer_limit=args.eval_batch_size)
    
    # Sampler
    sampler = Sampler()
    
    # Model
    model = DLPG(args=args).to(args.device)
    
    # Reward Normalizer
    if args.Running_Normalizer:
        normalizer = RunningNormalizer(decay=args.Running_Mean_decay)
    else:
        normalizer = WhiteningNormalizer()
    
    # Save args
    model.save_args()
    
    # Load Weight
    if args.LOAD_WEIGHTPATH is None:
        print("No Weights Found!! Starting From Scratch")
    elif Path.is_file(Path(args.LOAD_WEIGHTPATH)):
        model.load_state_dict(torch.load(args.LOAD_WEIGHTPATH),strict=True)
    else:
        raise FileNotFoundError()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr= args.lr)
    
    # Environment
    env = CustomReacherEnv(Kp=args.Kp, Kd=args.Kd, jointlimit=args.jointlimit)
    
    ### Loop ###
    from tqdm import tqdm
    pbar = tqdm(range(args.max_iterations))
    for iteration in pbar:
        # Fill train_buffer
        if args.Training:
            # select random quadrant to reset environment
            target_quadrant = np.random.randint(0,4)

            # DLPG generate trajectory
            EXPLORE_probabilty = 1/2**(iteration/args.exploit_iteration)
            EXPLOIT = np.random.rand() > EXPLORE_probabilty
            
            (anchor, rejection_rate), (reward,_,last_position,_) = model.rollout(env, target_quadrant, EXPLOIT, args.lentraj, args.RENDER_ROLLOUT)
            
            train_buffer.store(anchor, reward, target_quadrant, last_position)
            
            pbar.set_description(
                "Current iteration:{}, EXPLORE_probabilty:{:.2f}, Rejection_rate:{:.2f}".format(iteration+1,EXPLORE_probabilty, rejection_rate))
            
            if args.WANDB: wandb.log(
                {
                    'rejection_rate':rejection_rate, 
                    'EXPLOIT_probabilty':EXPLORE_probabilty
                    },step=iteration+1)

        # Update
        if args.Training:
            if (iteration+1)%args.update_every==0 and iteration > args.batch_size:
                ### Training ###
                model.train()
                # Sample
                if args.sample_method == "random":
                    batch = sampler.sample_random(buffer=train_buffer, batch_size = args.batch_size)
                elif args.sample_method == "LAdpp":
                    batch = sampler.sample_Frontier_LAdpp(train_buffer, test_buffer, test_points_number = len(test_buffer), batch_size = args.batch_size)
                else:
                    raise NameError(f"sample_method muse be one of ['random', 'LAdpp'], \n but found {args.sample_method}")
                
                train_log_dictionary = DLPG_Loss(model, batch, normalizer)
                
                optimizer.zero_grad()
                train_log_dictionary['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                if args.WANDB: wandb.log(prefix_dict(train_log_dictionary, prefix="train"),step=iteration+1)

        # Test
        if (iteration)%args.eval_batch_size==0:
            ### Evaluating ###
            model.eval()
            # Empty and Fill test_buffer
            test_buffer.empty()
            
            for _ in range(args.eval_batch_size):
                target_quadrant = np.random.randint(0,4)
                
                (anchor, _), (reward,_,last_position,_) = model.rollout(env, target_quadrant, True, args.lentraj, args.RENDER_ROLLOUT)
                
                test_buffer.store(anchor, reward, target_quadrant, last_position)
            
            batch = sampler.sample_all(buffer=test_buffer)
            
            
            test_log_dictionary = DLPG_Loss(model, batch, normalizer)
            measure_dict = get_measure(batch, args.mode, args.PLOT, plot_name = args.SAVE_RESULT_PATH/f"{iteration+1}.png" )
            
            test_log_dictionary.update(measure_dict)
            if args.WANDB: wandb.log(prefix_dict(test_log_dictionary, prefix="test"),step=iteration+1)
            
            if args.Running_Normalizer:
                normalizer.update(rewards=batch["reward"], quadrants=batch["target_quadrant"])
            
            print_log_dict(test_log_dictionary)
            
            if args.Training:
                model.save(iteration+1)    # save weight






if __name__ == "__main__":
    BASEDIR, RUNMODE = get_BASERDIR(__file__)
    


    # parser = argparse.ArgumentParser(description= 'parse for DLPG')
    # parser.add_argument("--model", type=str, default= "DLPG") # DLPG, DLPG_MDN, SAC, PPO
    # parser.add_argument("--configs", type=str, default= "../configs/MDN_DLPG/BASE.yaml")  
    
    # if RUNMODE in [DEBUG, JUPYTER] : 
    #     args, optional = parser.parse_known_args(
    #         args=["--WANDB", "False", "--Training", "True"])
    # else:
    #     args, optional = parser.parse_known_args()
    
    # if args.model == "DLPG":
    #     ARGS = Refine_OptionalArgs_to_Template(
    #         BASEDIR/args.configs, optional, DLPG_ARGS_Template)
    # elif args.model == "MDN_DLPG":
    #     ARGS = Refine_OptionalArgs_to_Template(
    #         BASEDIR/args.configs, optional, DLPG_ARGS_Template, cast_candidate = [torch.nn])

    # else: LookupError(f"args.model must be one of [DLPG, DLPG_MDN, SAC, PPO]\n. but found {args.model}")
    
    ARGS = DLPG_ARGS_Template()
    
    
    if RUNMODE is RUN:
        ARGS.WANDB = True
    else:
        ARGS.WANDB = False
    
    ARGS.runname = "DLPG_LAdpp"
    ARGS.sample_method = "LAdpp"
    main(ARGS)
    
    # ARGS.runname = "DLPG_random"
    # ARGS.sample_method = "random"
    # main(ARGS)


