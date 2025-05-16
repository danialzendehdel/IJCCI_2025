import os
from environment.create_env import create_env
from stable_baselines3.common.monitor import Monitor
import numpy as np

def make_env(path, journalist, rank, seed=0):
    """
    Creates a function that will initialize an environment when called.
    Wraps creation with error handling for better multiprocessing support.
    
    Args:
        path: Path to config file
        journalist: Journalist instance for logging (will be cloned for SubprocVecEnv)
        rank: Environment index for seeding
        seed: Base seed for randomization
    """
    def _init():
        try:
            # Set a process-specific numpy seed to avoid correlation between environments
            np.random.seed(seed + rank)
            
            # Create the base environment - pass a clone of journalist if needed
            env = create_env(path, journalist_ins=journalist)
            
            # Wrap with Monitor for logging
            env = Monitor(env)
            
            # Reset with appropriate seed
            env.reset(seed=seed + rank)
            
            return env
        except Exception as e:
            # Log the error but don't crash
            print(f"Error initializing environment {rank}: {e}")
            # Re-raise to ensure the problem is noticed
            raise
    
    return _init