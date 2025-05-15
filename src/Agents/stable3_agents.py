import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datetime
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
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
from environment.create_env import create_env
from utils.journal import Journal
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Episode tracking
        self.current_episode = 0
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        
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

def make_env(path, journalist, rank, seed=0):
    def _init():
        env = create_env(path, journalist_ins=journalist)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def setup_environment(path,journalist, num_envs=1):
    env = DummyVecEnv([make_env(path,journalist, i) for i in range(num_envs)])
    
    env = VecNormalize(
        env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-08
    )
    return env

def create_model(algo, env, device='cpu', **kwargs):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy_kwargs = dict(
        net_arch=dict(
            pi=[64, 64],  # Smaller network
            vf=[64, 64]
        ),
        activation_fn=torch.nn.ReLU,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=dict(eps=1e-5)
    )
    
    if algo.upper() == 'PPO':
        model = PPO(
            "MlpPolicy", 
            env,
            learning_rate=3e-4,
            n_steps=1024,      # Smaller batch size
            batch_size=64,     # Smaller minibatch
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device,
            tensorboard_log="Code/results/tensorboard_logs",
            **kwargs
        )
    elif algo.upper() == 'SAC':
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=1000000,
            learning_starts=100,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device,
            tensorboard_log="Code/results/tensorboard_logs",
            **kwargs
        )
    elif algo.upper() == 'TD3':
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=1000000,
            learning_starts=100,
            batch_size=100,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "episode"),
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device,
            tensorboard_log="Code/results/tensorboard_logs",
            **kwargs
        )
    return model

def setup_callbacks(env, save_freq=10000, eval_freq=10000):
    # Create directories if they don't exist
    os.makedirs("Code/results/checkpoints", exist_ok=True)
    os.makedirs("Code/results/best_model", exist_ok=True)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="Code/results/checkpoints",
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
        best_model_save_path="Code/results/best_model",
        verbose=1
    )
    
    # Custom tensorboard callback
    tensorboard_callback = TensorboardCallback()
    
    return CallbackList([checkpoint_callback, eval_callback, tensorboard_callback])

def train_and_evaluate(algo, total_timesteps, device="cpu"):
    try:
        # Create directories
        os.makedirs("Code/results/final_models", exist_ok=True)
        os.makedirs("Code/results/tensorboard_logs", exist_ok=True)
        
        # Get current time for unique naming
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')
        
        # Setup logging and environment
        journalist = Journal("Code/results/Training", "training" + formatted_time)
        env = setup_environment('src/configuration/config.yml', journalist)
        
        # Create callback
        tensorboard_callback = TensorboardCallback()
        checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path=f"Code/results/checkpoints/{algo}_{formatted_time}/",
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
                tensorboard_log=f"Code/results/tensorboard_logs",
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
            os.makedirs("Code/results/training_data", exist_ok=True)
            step_df.to_csv(f"Code/results/training_data/{algo}_steps_{formatted_time}.csv")
            episode_df.to_csv(f"Code/results/training_data/{algo}_episodes_{formatted_time}.csv")
        
        # Log training results
        journalist._process_smoothly(
            f"Training completed.\n"
            f"Final reward: {final_reward:.2f}\n"
            f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}"
        )
        
        # Evaluate the model
        try:
            eval_env = setup_environment('/home/danial/Documents/Codes_new/N-IJCCI-BMS/src/configuration/config.yml', journalist)
            eval_rewards = []
            n_eval_episodes = 5
            
            for i in range(n_eval_episodes):
                episode_reward = 0
                obs = eval_env.reset()[0]
                done = False
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = eval_env.step(action)
                    episode_reward += reward
                    done = done or truncated
                
                eval_rewards.append(episode_reward)
                journalist._process_smoothly(f"Evaluation episode {i+1}: Reward = {episode_reward:.2f}")
            
            mean_reward = np.mean(eval_rewards)
            std_reward = np.std(eval_rewards)
            journalist._process_smoothly(f"Evaluation complete. Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
            
        except Exception as e:
            journalist._get_error(f"Evaluation failed: {str(e)}")
            mean_reward, std_reward = 0.0, 0.0
        
        # Save final model
        model.save(f"Code/results/final_models/{algo}_final_{formatted_time}")
        journalist._process_smoothly(f"Model saved to Code/results/final_models/{algo}_final_{formatted_time}")
        
        return model, env
        
    except Exception as e:
        journalist._get_error(f"Training failed: {str(e)}")
        raise e

if __name__ == "__main__":
    print("Using CPU for training")
    device = "cpu"
    
    algorithms = ['PPO']
    data_length = 34700
    num_passes = 1
    total_steps = data_length * num_passes
    
    for algo in algorithms:
        print(f"\nTraining with {algo}")
        model, env = train_and_evaluate(
            algo=algo, 
            total_timesteps=total_steps,
            device=device
        )
        
        if device == "cuda":
            print(f"Final GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB") 