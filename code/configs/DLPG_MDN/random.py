from dataclasses import dataclass
from torch import tensor, Tensor

@dataclass
class ARGS():
    # MODEL
    model:str = "DLPG_MDN"
    
    # LOG
    WANDB:bool = True
    pname:str = "Reacher_Merge_v2"                      # WANDB project Name
    runname:str = "DLPG_MDN_random"                              # WANDB runname. If unspecified, set to datetime.
    
    # DATAIO
    LOAD_WEIGHTPATH:str = None                          # Path to load weight
    
    # Render
    RENDER_ROLLOUT:bool = False                         # Render Rollout with mujoco app
    RENDER_EVAL:bool = False                            # Plot Evalutation with plt

    # EVALUTATION
    eval_batch_size:int = 500                           # Rollout number when evaluating 
    mode: int = 4                                       # Number of Clusters to measure diversity
    PLOT:bool = True                                    # Setting True saves figure.

    
    # ENVIRONMENT
    jointlimit:Tensor = tensor([3.14, 2.22])            # Jointlimit of ReacherEnv
    position_dim:int = 2                                # Observed position state of ReacherEnv
    lentraj:int = 100                                   # Horizon of trajectory
    
    # TRAINING
    Training:bool = True                                # Training flag
    sample_method:str = "random"                        # 'random', 'LAdpp'
    
    exploit_iteration:int = 2_000_000                      # After exploit_iteration chance of exploration reduce to 50%
    max_iterations:int = 50_000                         # Maximum iterations
    
    buffer_limit: int = 10_000                          # Buffer size limit
    batch_size: int = 128                               # Batch size
    update_every:int = 100                              # Update after given number of epochs
    lr: float = 0.005                                   # learning rate
    
    # Device
    device:str = "cpu"                                  # device to use
    
    # Normalizer
    Running_Normalizer: bool = True
    
    # PD Controller
    Kp:float = 0.01*60                                  # Proprotional Gain of PD controller
    Kd:float = 0.001*60                                 # Dervative Gain of PD controller
    
    # ARCHITECTURE
    anchor_dim:int = 2
    cdim:int = 4
    zdim:int = 4
    eps_dim:int = 16
    hdim:tuple = (16,16)
    
    Running_Mean_decay:float = 0.95    
    n_components:int = 20
    
    # SEED
    random_seed:tuple = tuple(range(10))

