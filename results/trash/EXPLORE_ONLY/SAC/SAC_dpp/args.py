from dataclasses import dataclass
from torch import tensor, Tensor

@dataclass
class ARGS():
    # MODEL
    model:str = "SAC"
    
    # LOG
    WANDB:bool = True
    pname:str = "Reacher_Merge_v2.1"                      # WANDB project Name
    runname:str = "SAC_dpp"                              # WANDB runname. If unspecified, set to datetime.
    
    # DATAIO
    LOAD_WEIGHTPATH:str = None                          # Path to load weight
    
    # Render
    RENDER_ROLLOUT:bool = False                         # Render Rollout with mujoco app
    RENDER_EVAL:bool = False                            # Plot Evalutation with plt

    # EVALUTATION
    eval_batch_size:int = 500                           # Rollout number when evaluating 
    mode: int = 4                                       # Number of Clusters to measure diversity
    PLOT:bool = True                                    # Setting True saves figure.
    test_points_ratio:float = 0.0                       # test_points_ratio for LAdpp sampling
    
    # ENVIRONMENT
    jointlimit:Tensor = tensor([3.14, 2.22])            # Jointlimit of ReacherEnv
    position_dim:int = 2                                # Observed position state of ReacherEnv
    lentraj:int = 100                                   # Horizon of trajectory
    
    # TRAINING
    Training:bool = True                                # Training flag
    sample_method:str = "LAdpp"                        # 'random', 'LAdpp'
    
    max_iterations:int = 50_000                         # Maximum iterations
    
    buffer_limit: int = 10_000                          # Buffer size limit
    batch_size: int = 128                               # Batch size
    update_every:int = 100                              # Update after given number of epochs
    lr: float = 0.005                                   # learning rat
    tau: float = 0.005                                  # decay rate of target network
    init_alpha:float = 0.1                              # Entropy Scale Term
    lr_alpha:float = 3e-4                               # learning rate of alpha
    
    # Device
    device:str = "cpu"                                  # device to use
    
    # PD Controller
    Kp:float = 0.01*60                                  # Proprotional Gain of PD controller
    Kd:float = 0.001*60                                 # Dervative Gain of PD controller
    
    # ARCHITECTURE
    anchor_dim:int = 2
    cdim:int = 4
    zdim:int = 4
    eps_dim:int = 16
    hdim:tuple = (16,16)
    
    # SEED
    random_seed:tuple = tuple(range(10))

