import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datetime
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback, 
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
import torch
from src.environment.create_env import create_env
from utils.journal import Journal
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils_training.plot_csv import plot_results
import datetime
from utils.utils_training.callbacks import TensorboardCallback
from utils.utils_training.create_models import create_model
from utils.utils_training.setup_envs import make_env
import multiprocessing



def setup_environment(path, journalist, num_envs=None):
    # Automatically determine optimal number of environments based on CPU cores
    if num_envs is None:
        num_cpus = multiprocessing.cpu_count()
        # Use fewer environments than available cores to prevent resource exhaustion
        # This is more conservative than before to address the broken pipe error
        num_envs = max(16, (num_cpus // 4))  # Using at most 4 environments or 1/4 of cores
        journalist._process_smoothly(f"Using {num_envs} parallel environments (conservative setting)")
    
    # For more reliability, try SubprocVecEnv but fall back to DummyVecEnv if needed
    try:
        journalist._process_smoothly(f"Initializing SubprocVecEnv with {num_envs} workers...")
        env = SubprocVecEnv([make_env(path, journalist, i) for i in range(num_envs)])
        journalist._process_smoothly("Successfully created SubprocVecEnv")
    except Exception as e:
        journalist._get_warning(f"Failed to create SubprocVecEnv: {e}. Falling back to DummyVecEnv.")
        # Fall back to DummyVecEnv if SubprocVecEnv fails
        env = DummyVecEnv([make_env(path, journalist, 0)])
    
    env = VecNormalize(
        env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-08
    )
    return env


def configure_cpu_threads():
    """
    Configure PyTorch to use all available CPU cores.
    Returns the number of cores available.
    """
    # Get number of available CPU cores
    num_cores = multiprocessing.cpu_count()
    
    # Configure PyTorch to use all cores - do this safely
    try:
        # Set PyTorch thread count
        torch.set_num_threads(num_cores)
        
        # Only try to set interop threads if it hasn't been initialized
        if not hasattr(torch, '_initialized_interop_threads'):
            torch.set_num_interop_threads(num_cores)
            setattr(torch, '_initialized_interop_threads', True)
    except Exception as e:
        print(f"Warning: Could not set all PyTorch thread options: {e}")
        print(f"Continuing with default thread configuration")
    
    # Set OpenMP threads if available (used by some PyTorch operations)
    try:
        os.environ["OMP_NUM_THREADS"] = str(num_cores)
        os.environ["MKL_NUM_THREADS"] = str(num_cores)
    except Exception:
        pass
    
    return num_cores


def train_and_evaluate(algo, total_timesteps, device="cpu", formatted_time=None):
    # Get current time for unique naming
    current_time = datetime.datetime.now()
    formatted_time = formatted_time or current_time.strftime('%Y-%m-%d_%H-%M-%S')
    
    # Setup logging at the beginning, outside the try block
    journalist = Journal("results/Training", "training" + formatted_time)
    
    try:
        # Configure CPU usage
        num_cores = configure_cpu_threads()
        journalist._process_smoothly(f"Using {num_cores} CPU cores for computation")
        
        # Create directories
        os.makedirs("results/final_models", exist_ok=True)
        os.makedirs("results/tensorboard_logs", exist_ok=True)
        os.makedirs("results/csvs", exist_ok=True)
        
        # Setup environment
        env = setup_environment('src/configuration/config.yml', journalist)
        
        # Create callback with CSV path
        tensorboard_callback = TensorboardCallback(
            csv_path="results/csvs",
            log_freq=1,
            formatted_time=formatted_time
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path=f"results/checkpoints/{algo}_{formatted_time}/",
            name_prefix="rl_model"
        )
        
        # Combine callbacks
        callbacks = CallbackList([tensorboard_callback, checkpoint_callback])
        
        # Create and train model
        if algo == "PPO":
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                device=device,
                tensorboard_log=f"results/tensorboard_logs",
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")
        
        journalist._process_smoothly(f"{algo} model created successfully")
        journalist._process_smoothly("Starting training...")
        
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=f"{algo}",
            progress_bar=True
        )
        
        # Get training data
        step_df, episode_df = tensorboard_callback.get_data()
        
        # Handle case where no steps were completed
        if step_df is None or episode_df is None:
            journalist._get_warning("No steps completed during training")
            final_reward = 0
            mean_reward = 0
            std_reward = 0
        else:
            # Calculate statistics from completed steps
            final_reward = step_df['reward'].iloc[-1] if not step_df.empty else 0
            mean_reward = step_df['reward'].mean() if not step_df.empty else 0
            std_reward = step_df['reward'].std() if not step_df.empty else 0
            
            # Save training data
            os.makedirs("results/training_data", exist_ok=True)
            step_df.to_csv(f"results/training_data/{algo}_steps_{formatted_time}.csv")
            episode_df.to_csv(f"results/training_data/{algo}_episodes_{formatted_time}.csv")
        
        # Log training results
        journalist._process_smoothly(
            f"Training completed.\n"
            f"Final reward: {final_reward:.2f}\n"
            f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}"
        )
        
        # Generate plots from the CSV data
        csv_file = f"results/csvs/training_data.csv"
        if os.path.exists(csv_file):
            plot_results(csv_file, formatted_time)
            journalist._process_smoothly(f"Plots generated from CSV data")
        
        # Evaluate the model
        try:
            config_path = 'src/configuration/config.yml'
            journalist._process_smoothly(f"Creating evaluation environment with config: {config_path}")
            
            # For evaluation, use a single environment to simplify evaluation logic
            eval_env = setup_environment(config_path, journalist, num_envs=1)
            eval_rewards = []
            n_eval_episodes = 2  # Reduce number of evaluation episodes to save time
            
            journalist._process_smoothly(f"Starting evaluation for {n_eval_episodes} episodes")
            
            for i in range(n_eval_episodes):
                episode_reward = 0
                # VecEnv environments return a tuple with the first element being the observation
                journalist._process_smoothly(f"Resetting environment for evaluation episode {i+1}")
                try:
                    obs = eval_env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]  # Extract observation from tuple if needed
                except Exception as reset_error:
                    journalist._get_error(f"Error resetting environment: {reset_error}")
                    break
                    
                done = False
                
                # Add step counter and limit to prevent infinite loops
                step_count = 0
                max_eval_steps = 1000  # Prevent infinite evaluation
                
                while not done and step_count < max_eval_steps:
                    step_count += 1
                    if step_count % 100 == 0:
                        journalist._process_smoothly(f"Evaluation step {step_count} in episode {i+1}")
                        
                    try:
                        action, _ = model.predict(obs, deterministic=True)
                        # Step the environment
                        result = eval_env.step(action)
                        
                        # Handle different return value structures
                        if len(result) == 4:  # Old format or Vec wrapped
                            obs, reward, done_arr, info = result
                            done = done_arr[0] if isinstance(done_arr, (list, np.ndarray)) else done_arr
                        else:  # New gymnasium format through VecEnv
                            obs, reward, terminated, truncated, info = result
                            done = terminated[0] or truncated[0] if isinstance(terminated, (list, np.ndarray)) else terminated or truncated
                            
                        reward = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
                        episode_reward += reward
                        
                    except Exception as step_error:
                        journalist._get_error(f"Error during evaluation step: {step_error}")
                        done = True  # End this episode
                
                # Log if max steps reached
                if step_count >= max_eval_steps:
                    journalist._get_warning(f"Evaluation episode {i+1} reached step limit without finishing.")
                    
                eval_rewards.append(episode_reward)
                journalist._process_smoothly(f"Evaluation episode {i+1}: Reward = {episode_reward:.2f}")
            
            if eval_rewards:
                mean_reward = np.mean(eval_rewards)
                std_reward = np.std(eval_rewards)
                journalist._process_smoothly(f"Evaluation complete. Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
            else:
                journalist._get_warning("No evaluation episodes completed successfully")
                mean_reward, std_reward = 0.0, 0.0
            
        except Exception as e:
            journalist._get_error(f"Evaluation failed: {str(e)}")
            mean_reward, std_reward = 0.0, 0.0
        
        # Save final model
        model.save(f"results/final_models/{algo}_final_{formatted_time}")
        journalist._process_smoothly(f"Model saved to results/final_models/{algo}_final_{formatted_time}")
        
        return model, env
        
    except Exception as e:
        journalist._get_error(f"Training failed: {str(e)}")
        raise e

if __name__ == "__main__":
    # Configure CPU usage at the start
    num_cores = configure_cpu_threads()
    print(f"Using {num_cores} CPU cores for PyTorch operations")
    print("Using CPU for training")
    
    # Set more conservative memory usage for multiprocessing
    os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads per process
    os.environ["MKL_NUM_THREADS"] = "1"  # Limit MKL threads per process
    
    # Explicitly set start method to spawn for better isolation
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # Method already set, ignore error
        pass
    
    # Set training parameters
    device = "cpu"
    algorithms = ['PPO']
    
    # Reduce total steps for testing if needed
    data_length = 346964
    num_passes = 1
    
    # Set to full training length
    total_steps = data_length * num_passes  # Full training
    # total_steps = 5000  # Start with a small number to verify stability

    formatted_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    for algo in algorithms:
        print(f"\nTraining with {algo}")
        try:
            model, env = train_and_evaluate(
                algo=algo, 
                total_timesteps=total_steps,
                device=device,
                formatted_time=formatted_time
            )
        except Exception as e:
            print(f"Training failed with error: {e}")
            # Clean up any leftover processes
            for process in multiprocessing.active_children():
                print(f"Terminating leftover process: {process.name}")
                process.terminate()
            
        if device == "cuda":
            print(f"Final GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB") 