# %%
from pathlib import Path
import sys
import argparse
import wandb
from utils.dataloader import Buffer, Sampler
from model.DLPG import DLPG
from model.DLPG_MDN import DLPG_MDN
from model.SAC import SAC
import os

from envs.reacherEnv import CustomReacherEnv
import numpy as np
import random
import torch

from configs.template import DLPG_ARGS_Template, DLPG_MDN_ARGS_Template, SAC_ARGS_Template
from typing import Union

from utils.tools import set_seed, set_wandb, print_log_dict, prefix_dict
from utils.path_handler import JUPYTER, RUN, DEBUG, get_BASERDIR
from model.Normalizer import RunningNormalizer, WhiteningNormalizer
from utils.update import DLPG_update, DLPG_MDN_update, SAC_update
from utils.args import read_ARGS
from utils.measure import get_measure
from utils.logger import CSVLogger,ask_and_make_folder
from utils.tools import cast_numpy

print(sys.version)


def main(args: Union[DLPG_ARGS_Template, DLPG_MDN_ARGS_Template, SAC_ARGS_Template], seed):
    ### SEED ###
    set_seed(seed)
    
    ### WANDB ### 
    if args.WANDB:
        set_wandb(pname = args.pname, runname = args.runname)
    
    ### Logger ###
    csvlogger = CSVLogger(args.SAVE_RESULT_PATH/f"test_result_seed_{seed}.csv")
    
    ### Declare Instance ###
    # Buffer
    train_buffer = Buffer(
        anchor_dim=args.anchor_dim, last_position_dim=args.position_dim, buffer_limit=args.buffer_limit, )
    
    test_buffer = Buffer(
        anchor_dim=args.anchor_dim, last_position_dim=args.position_dim, buffer_limit=args.eval_batch_size)
    
    # Sampler
    sampler = Sampler()
    
    # Model
    if args.model == "DLPG":
        model = DLPG(args=args).to(args.device)
    elif args.model == "DLPG_MDN":
        model = DLPG_MDN(args=args).to(args.device)
    elif args.model == "SAC":
        model = SAC(args=args).to(args.device)
    else:
        raise LookupError(f"model should be one of ['DLPG', 'DLPG_MDN', 'SAC', 'PPO'] \n, Found {args.model}")
    
    # Reward Normalizer
    if args.model in ["DLPG", "DLPG_MDN"]:
        # Reward Normalizer
        if args.Running_Normalizer:
            normalizer = RunningNormalizer(decay=args.Running_Mean_decay)
        else:
            normalizer = WhiteningNormalizer()
    
    # Load Weight
    if args.LOAD_WEIGHTPATH is None:
        print("No Weights Found!! Starting From Scratch")
    elif Path.is_file(Path(args.LOAD_WEIGHTPATH)):
        model.load_state_dict(torch.load(args.LOAD_WEIGHTPATH),strict=True)
    else:
        raise FileNotFoundError()
    
    # Environment
    env = CustomReacherEnv(Kp=args.Kp, Kd=args.Kd, jointlimit=cast_numpy(args.jointlimit))
    
    ### Loop ###
    from tqdm import tqdm
    pbar = tqdm(range(args.max_iterations+1))
    for iteration in pbar:
        # Fill train_buffer
        if args.Training:
            # select random quadrant to reset environment
            target_quadrant = np.random.randint(0,4)

            # ROLL OUT
            if args.model in ["DLPG", "DLPG_MDN"]:
                EXPLORE_probabilty = 1/2**(iteration/args.exploit_iteration)
                EXPLOIT = np.random.rand() > EXPLORE_probabilty
                
                (anchor, rejection_rate), (reward,_,last_position,_) = model.rollout(env, target_quadrant, EXPLOIT, args.lentraj, args.RENDER_ROLLOUT)
                if args.WANDB: 
                    wandb.log({'rejection_rate':rejection_rate, 'EXPLOIT_probabilty':EXPLORE_probabilty},step=iteration+1)
                
            elif args.model == "SAC":
                (anchor, rejection_rate), (reward,_,last_position,_) = model.rollout(env, target_quadrant, args.lentraj, args.RENDER_ROLLOUT)
                if args.WANDB: 
                    wandb.log({'rejection_rate':rejection_rate},step=iteration+1)
                    
            else:
                raise LookupError(f"model should be one of ['DLPG', 'DLPG_MDN', 'SAC', 'PPO'] \n, Found {args.model}")
                
            pbar.set_description(
                "Current iteration:{},  Rejection_rate:{:.2f}".format(iteration+1, rejection_rate))
            train_buffer.store(anchor, reward, target_quadrant, last_position)
            
            

        # Update
        if args.Training:
            if (iteration+1)%args.update_every==0 and iteration > args.batch_size:
                ### Training ###
                model.train()
                
                # Sample
                if args.sample_method == "random":
                    batch = sampler.sample_random(buffer=train_buffer, batch_size = args.batch_size)
                elif args.sample_method == "LAdpp":
                    batch = sampler.sample_Frontier_LAdpp(train_buffer, test_buffer, test_points_ratio = args.test_points_ratio, batch_size = args.batch_size)
                else:
                    raise NameError(f"sample_method muse be one of ['random', 'LAdpp'], \n but found {args.sample_method}")
                
                # Update
                if args.model == "DLPG":
                    train_log_dictionary = DLPG_update(model, batch, normalizer, TRAIN=True)
                elif args.model == "DLPG_MDN":
                    train_log_dictionary = DLPG_MDN_update(model, batch, normalizer, TRAIN=True)
                elif args.model == "SAC":
                    train_log_dictionary = SAC_update(model, batch, TRAIN=True)
                else:
                    raise LookupError(f"model should be one of ['DLPG', 'DLPG_MDN', 'SAC', 'PPO'] \n, Found {args.model}")
                
                
                if args.WANDB: wandb.log(prefix_dict(train_log_dictionary, prefix="train"),step=iteration+1)

        # Test
        if (iteration)%args.eval_batch_size==0:
            ### Evaluating ###
            model.eval()
            # Empty and Fill test_buffer
            test_buffer.empty()
            
            for _ in range(args.eval_batch_size):
                target_quadrant = np.random.randint(0,4)
                
                # ROLL OUT
                if args.model in ["DLPG", "DLPG_MDN"]:
                    (anchor, rejection_rate), (reward,_,last_position,_) = model.rollout(env, target_quadrant, True, args.lentraj, args.RENDER_ROLLOUT)
                    if args.WANDB: 
                        wandb.log({'rejection_rate':rejection_rate, 'EXPLOIT_probabilty':EXPLORE_probabilty},step=iteration+1)
                    
                elif args.model == "SAC":
                    (anchor, rejection_rate), (reward,_,last_position,_) = model.rollout(env, target_quadrant, args.lentraj, args.RENDER_ROLLOUT)
                    if args.WANDB: 
                        wandb.log({'rejection_rate':rejection_rate},step=iteration+1)
                        
                else:
                    raise LookupError(f"model should be one of ['DLPG', 'DLPG_MDN', 'SAC', 'PPO'] \n, Found {args.model}")
                
                test_buffer.store(anchor, reward, target_quadrant, last_position)
            
            batch = sampler.sample_all(buffer=test_buffer)
            
            # Get log_dictionary
            if args.model == "DLPG":
                test_log_dictionary = DLPG_update(model, batch, normalizer, TRAIN=False)
            elif args.model == "DLPG_MDN":
                test_log_dictionary = DLPG_MDN_update(model, batch, normalizer, TRAIN=False)
            elif args.model == "SAC":
                test_log_dictionary = SAC_update(model, batch, TRAIN=False)
            else:
                raise LookupError(f"model should be one of ['DLPG', 'DLPG_MDN', 'SAC', 'PPO'] \n, Found {args.model}")
            
            # Get measure
            measure_dict = get_measure(batch, args.mode, args.PLOT, plot_name = args.SAVE_RESULT_PATH/"plots"/f"{iteration+1}.png" )
            
            test_log_dictionary.update(measure_dict)
            if args.WANDB: wandb.log(prefix_dict(test_log_dictionary, prefix="test"),step=iteration+1)
            
            # Update Normalizer
            if args.model in ["DLPG", "DLPG_MDN"]:
                if args.Running_Normalizer:
                    normalizer.update(rewards=batch["reward"], quadrants=batch["target_quadrant"])
                    if args.WANDB: wandb.log({"Reward_Thres":normalizer.get_buffer("Running_mean").mean()},step=iteration+1)
                
            csvlogger.log(test_log_dictionary)
            print_log_dict(test_log_dictionary)
            
            if args.Training:
                model.save(iteration+1)    # save weight



if __name__ == "__main__":
    BASEDIR, RUNMODE = get_BASERDIR(__file__)

    parser = argparse.ArgumentParser(description= 'parse for DLPG')
    parser.add_argument("--configs", type=str) # [DLPG, DLPG_MDN, SAC, PPO], [random, dpp, Fdpp]
    args= parser.parse_args()

    ARGS = read_ARGS((BASEDIR/'configs'/args.configs).absolute())
    
    ### Save ARGS ###
    import shutil
    SAVEPATH = Path(BASEDIR).parent/"results"/ARGS.model/ARGS.runname
    
    ask_and_make_folder(SAVEPATH)
    shutil.copyfile((BASEDIR/'configs'/args.configs).absolute(), SAVEPATH/"args.py")
    
    for idx, seed in enumerate(ARGS.random_seed):
        if seed == 0 and (RUNMODE is RUN) and ARGS.WANDB:
            ARGS.WANDB = True
        else:
            ARGS.WANDB = False
        
        ### Set Path ###
        ARGS.SAVE_RESULT_PATH = Path(BASEDIR).parent/"results"/ARGS.model/ARGS.runname/f"seed_{seed}"
        ARGS.SAVE_WEIGHT_PATH = Path(BASEDIR).parent/"results"/ARGS.model/ARGS.runname/f"seed_{seed}"/"weights"

        Path.mkdir(ARGS.SAVE_WEIGHT_PATH,exist_ok=True, parents=True)
        Path.mkdir(ARGS.SAVE_RESULT_PATH,exist_ok=True, parents=True)
        
        main(ARGS, seed)


