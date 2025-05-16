from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback, 
    CallbackList,
    BaseCallback
)
import pandas as pd
import numpy as np
import os

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from utils.utils_training.setup_envs import make_env

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0, csv_path=None, log_freq=1, formatted_time=None):
        super().__init__(verbose)
        # Episode tracking
        self.current_episode = 0
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        self.csv_path = csv_path or "results/csvs"
        self.log_freq = log_freq  # How often to save CSV data (every n steps)
        self.formatted_time = formatted_time
        
        # History tracking for all info metrics
        self.history = {
            'steps': {
                'soc_value': [],
                'soc_violation': [],
                'soc_clipped': [],
                'current': [],
                'resistance': [],
                'action_value': [],
                'action_violation': [],
                'action_clipped': [],
                'p_loss': [],
                'q_loss': [],
                'q_loss_percent': [],
                'throughput_charge': [],
                'reward': [],
                'soh': [],
                'ah_total': [],
                'n_val': [],
                'profit_cost': [],
                'profit_cost_without_battery': [],
                'cost': [],
                'battery_wear_cost': [],
                'total_cycle': [],
                'accumulated_reward': [],
                'accumulated_cost': [],
                'accumulated_battery_wear_cost': []
            },
            'episodes': {
                'total_reward': [],
                'length': [],
                'final_soc': [],
                'mean_soc': [],
                'final_q_loss_percent': [],
                'total_p_loss': [],
                'total_q_loss': [],
                'soc_violations': [],
                'action_violations': [],
                'final_throughput': [],
                'final_soh': [],
                'final_ah_total': [],
                'final_n_val': [],
                'total_cost': [],
                'total_battery_wear_cost': [],
                'total_profit': [],
                'total_cycle': [],
                'accumulated_reward': [],
                'accumulated_cost': [],
                'accumulated_battery_wear_cost': []
            }
        }
        
    def _on_step(self):
        info = self.locals['infos'][0]
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]
        
        # Save to CSV every log_freq steps, but handle different env structures
        if self.num_timesteps % self.log_freq == 0:
            try:
                # Different approach to save CSV based on environment type
                # For SubprocVecEnv, we need to use a remote call
                if hasattr(self.training_env, 'env_method'):
                    # Call the append_to_csv method on the first environment
                    self.training_env.env_method(
                        'append_to_csv',
                        [f"{self.csv_path}/training_data.csv"],
                        indices=[0]
                    )
                # Fall back to direct access if env_method is not available
                elif hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                    env = self.training_env.envs[0].env
                    env.append_to_csv(filepath=f"{self.csv_path}/training_data.csv")
                else:
                    print("Warning: Could not save CSV - environment structure not recognized")
            except Exception as e:
                print(f"Error saving CSV: {e}")
        
        # Initialize step values
        step_values = {key: None for key in self.history['steps'].keys()}
        
        # Log every step metric
        for key in info:
            if key in self.history['steps']:
                # Safely get the value, handling both list and non-list cases
                if isinstance(info[key], list):
                    if len(info[key]) > 0:  # Check if list is not empty
                        step_values[key] = info[key][-1]
                else:
                    step_values[key] = info[key]
        
        # Add reward to step values
        step_values['reward'] = reward
        
        # Update history only if we have values
        for key, value in step_values.items():
            if value is not None:
                self.history['steps'][key].append(value)
                self.logger.record(f"steps/{key}", value)
        
        self.current_episode_reward += reward
        self.current_episode_steps += 1
        
        # If episode ended
        if done:
            # Calculate episode statistics
            episode_stats = {
                'total_reward': self.current_episode_reward,
                'length': self.current_episode_steps
            }
            
            # Safely add other episode stats
            if len(self.history['steps']['soc_value']) >= self.current_episode_steps:
                episode_stats.update({
                    'final_soc': self.history['steps']['soc_value'][-1],
                    'mean_soc': np.mean(self.history['steps']['soc_value'][-self.current_episode_steps:])
                })
            
            # Add other metrics safely
            metric_mapping = {
                'throughput_charge': 'final_throughput',
                'soh': 'final_soh',
                'ah_total': 'final_ah_total',
                'n_val': 'final_n_val',
                'q_loss_percent': 'final_q_loss_percent'
            }
            
            for step_key, episode_key in metric_mapping.items():
                if step_key in self.history['steps'] and len(self.history['steps'][step_key]) > 0:
                    episode_stats[episode_key] = self.history['steps'][step_key][-1]
            
            # Calculate totals and violations
            if len(self.history['steps']['p_loss']) >= self.current_episode_steps:
                episode_stats['total_p_loss'] = np.sum(self.history['steps']['p_loss'][-self.current_episode_steps:])
            
            if len(self.history['steps']['q_loss']) >= self.current_episode_steps:
                episode_stats['total_q_loss'] = np.sum(self.history['steps']['q_loss'][-self.current_episode_steps:])
            
            if len(self.history['steps']['soc_violation']) >= self.current_episode_steps:
                episode_stats['soc_violations'] = sum(self.history['steps']['soc_violation'][-self.current_episode_steps:])
            
            if len(self.history['steps']['action_violation']) >= self.current_episode_steps:
                episode_stats['action_violations'] = sum(self.history['steps']['action_violation'][-self.current_episode_steps:])
                
            if len(self.history['steps']['cost']) >= self.current_episode_steps:
                episode_stats['total_cost'] = np.sum(self.history['steps']['cost'][-self.current_episode_steps:])
                
            if len(self.history['steps']['battery_wear_cost']) >= self.current_episode_steps:
                episode_stats['total_battery_wear_cost'] = np.sum(self.history['steps']['battery_wear_cost'][-self.current_episode_steps:])
                
            if len(self.history['steps']['profit_cost']) >= self.current_episode_steps:
                episode_stats['total_profit'] = np.sum(self.history['steps']['profit_cost'][-self.current_episode_steps:])
            
            # Track accumulated metrics at the episode level (use the final values)
            for accumulated_metric in ['accumulated_reward', 'accumulated_cost', 'accumulated_battery_wear_cost', 'total_cycle']:
                if accumulated_metric in self.history['steps'] and len(self.history['steps'][accumulated_metric]) >= self.current_episode_steps:
                    episode_stats[accumulated_metric] = self.history['steps'][accumulated_metric][-1]
            
            # Store episode stats in history
            for key, value in episode_stats.items():
                if value is not None and key in self.history['episodes']:
                    self.history['episodes'][key].append(value)
                    self.logger.record(f"episode/{key}", value)
            
            # Reset episode counters
            self.current_episode += 1
            self.current_episode_reward = 0
            self.current_episode_steps = 0
            
            # Dump logs
            self.logger.dump(self.num_timesteps)
        
        return True

    def get_data(self):
        """Return both step and episode data"""
        if not self.history['steps']['reward']:  # If no steps completed
            return None, None
        
        # Ensure all arrays have the same length for step data
        keys_to_include = []
        min_length = float('inf')
        
        for key, arr in self.history['steps'].items():
            if arr:  # If the array is not empty
                min_length = min(min_length, len(arr))
                keys_to_include.append(key)
        
        if min_length == float('inf'):
            return None, None
            
        # Truncate all arrays to minimum length
        step_data = {
            key: self.history['steps'][key][:min_length] for key in keys_to_include
        }
        
        # Create DataFrames
        step_df = pd.DataFrame(step_data)
        step_df['step'] = range(len(step_df))
        
        episode_keys = [k for k, v in self.history['episodes'].items() if v]
        episode_data = {
            key: self.history['episodes'][key] for key in episode_keys
        }
        
        episode_df = pd.DataFrame(episode_data)
        episode_df['episode'] = range(len(episode_df))
        
        return step_df, episode_df



def setup_callbacks(env, save_freq=10000, eval_freq=10000):
    # Create directories if they don't exist
    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/best_model", exist_ok=True)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="results/checkpoints",
        name_prefix="rl_model"
    )
    
    # Custom eval callback that handles evaluation
    class SimpleEvalCallback(BaseCallback):
        def __init__(self, eval_freq=10000, best_model_save_path=None, verbose=1):
            super().__init__(verbose)
            self.eval_freq = eval_freq
            self.best_mean_reward = -np.inf
            self.best_model_save_path = best_model_save_path
            
        def _on_step(self):
            # Skip evaluation if we're near the end of the dataset
            if self.n_calls >= 9900:  # Skip evaluation near the end
                return True
                
            if self.n_calls % self.eval_freq == 0:
                try:
                    # Create a fresh evaluation environment
                    eval_env = DummyVecEnv([make_env('src/configuration/config.yml', 0, seed=42)])
                    eval_env = VecNormalize(
                        eval_env,
                        norm_obs=False,
                        norm_reward=True,
                        clip_reward=10.0,
                        gamma=0.99,
                        epsilon=1e-08,
                        training=False
                    )
                    
                    # Run shorter evaluation episodes
                    episode_rewards = []
                    for _ in range(3):  # Reduced number of evaluation episodes
                        obs = eval_env.reset()
                        done = False
                        episode_reward = 0.0
                        step_count = 0
                        
                        while not done and step_count < 1000:  # Add step limit
                            action, _ = self.model.predict(obs, deterministic=True)
                            obs, reward, done, _ = eval_env.step(action)
                            episode_reward += reward
                            step_count += 1
                            
                        episode_rewards.append(episode_reward)
                    
                    mean_reward = np.mean(episode_rewards)
                    std_reward = np.std(episode_rewards)
                    
                    # Logging
                    self.logger.record("eval/mean_reward", mean_reward)
                    self.logger.record("eval/std_reward", std_reward)
                    
                    # Save best model
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        if self.best_model_save_path is not None:
                            path = os.path.join(self.best_model_save_path, "best_model")
                            self.model.save(path)
                    
                except Exception as e:
                    print(f"Evaluation failed: {e}")
                finally:
                    if 'eval_env' in locals():
                        eval_env.close()
            
            return True
    
    eval_callback = SimpleEvalCallback(
        eval_freq=eval_freq,
        best_model_save_path="results/best_model",
        verbose=1
    )
    
    # Custom tensorboard callback
    tensorboard_callback = TensorboardCallback()
    
    return CallbackList([checkpoint_callback, eval_callback, tensorboard_callback])



