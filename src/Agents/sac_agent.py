import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datetime
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback, 
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from environment.create_env import create_env
from utils.journal import Journal
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.utils import set_random_seed
from utils.config_reader import load_config

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
                'total_profit': []
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

class ResetTrackingCallback(BaseCallback):
    def __init__(self, config, verbose=0):
        super().__init__(verbose)
        self.resets_due_to_battery_waste = 0
        self.resets_due_to_data_completion = 0
        self.last_infos = None
        self.battery_eol_timestamps = []  # Store timestamps when batteries reach EOL
        self.sim_start_time = None        # Track simulation start time
        self.total_sim_hours = 0          # Track total simulation time in hours
        self.last_soh_values = []         # Track SOH values
        self.episode_count = 0            # Track episodes
        self.config = config              # Store config object
        
    def _on_step(self):
        # Check if episode ended
        if any(self.locals['dones']):
            for i, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][i]
                    # Check if episode ended due to battery waste or data completion
                    if 'terminal_observation' in info:
                        # Handle the case where soh might be a list or a float
                        soh_value = info.get('soh', 1.0)
                        if isinstance(soh_value, list):
                            soh_value = soh_value[-1] if soh_value else 1.0
                        
                        if soh_value <= self.config.simulation.soh_min:
                            self.resets_due_to_battery_waste += 1
                            self.logger.record("reset/battery_waste", self.resets_due_to_battery_waste)
                            
                            # Track time when battery reached EOL
                            curr_steps = self.model.num_timesteps if self.model else self.num_timesteps
                            time_interval = self.config.simulation.time_interval  # time step in hours
                            curr_sim_hours = curr_steps * time_interval
                            self.battery_eol_timestamps.append(curr_sim_hours)
                            self.logger.record("battery/eol_hours", curr_sim_hours)
                            
                            # Calculate and log average battery lifetime
                            if len(self.battery_eol_timestamps) > 1:
                                avg_lifetime_hours = self.calculate_avg_lifetime()
                                self.logger.record("battery/avg_lifetime_hours", avg_lifetime_hours)
                                self.logger.record("battery/avg_lifetime_years", avg_lifetime_hours / (24 * 365))
                                self.logger.record("battery/estimated_replacements_10years", 10 * 365 * 24 / avg_lifetime_hours)
                        else:
                            self.resets_due_to_data_completion += 1
                            self.logger.record("reset/data_completion", self.resets_due_to_data_completion)
                    
                    # Track episode count
                    self.episode_count += 1
                    
                    # Store SOH for degradation rate calculation
                    soh_value = info.get('soh', None)
                    if soh_value is not None:
                        if isinstance(soh_value, list):
                            soh_value = soh_value[-1] if soh_value else None
                        if soh_value:
                            self.last_soh_values.append(soh_value)
                            # Only keep last 10 values for calculation
                            if len(self.last_soh_values) > 10:
                                self.last_soh_values.pop(0)
                            
                            # Calculate degradation rate if we have enough data
                            if len(self.last_soh_values) >= 2:
                                hours_per_episode = self.config.environment.max_steps_per_episode * self.config.simulation.time_interval
                                degradation_rate = self.calculate_degradation_rate(hours_per_episode)
                                self.logger.record("battery/degradation_rate_per_hour", degradation_rate)
                                
                                # Project time to reach EOL
                                if degradation_rate > 0:
                                    time_to_eol_hours = self.project_time_to_eol(degradation_rate)
                                    self.logger.record("battery/projected_time_to_eol_hours", time_to_eol_hours)
                                    self.logger.record("battery/projected_time_to_eol_years", time_to_eol_hours / (24 * 365))
                    
            # Update total simulation time
            if self.model:
                time_interval = self.config.simulation.time_interval  # time step in hours
                self.total_sim_hours = self.model.num_timesteps * time_interval
                self.logger.record("simulation/total_hours", self.total_sim_hours)
                self.logger.record("simulation/total_years", self.total_sim_hours / (24 * 365))
        
        return True
    
    def calculate_avg_lifetime(self):
        """Calculate average battery lifetime in hours"""
        if len(self.battery_eol_timestamps) <= 1:
            return self.battery_eol_timestamps[0] if self.battery_eol_timestamps else 0
            
        # Calculate differences between consecutive replacements
        lifetimes = []
        prev_time = 0
        for timestamp in self.battery_eol_timestamps:
            lifetime = timestamp - prev_time
            lifetimes.append(lifetime)
            prev_time = timestamp
            
        # Return average lifetime (excluding first if there are many)
        if len(lifetimes) > 2:
            return sum(lifetimes[1:]) / len(lifetimes[1:])  # Skip first which might be partial
        return sum(lifetimes) / len(lifetimes)
    
    def calculate_degradation_rate(self, hours_per_episode):
        """Calculate battery degradation rate per hour"""
        if len(self.last_soh_values) < 2:
            return 0
            
        # Calculate average degradation over recent episodes
        total_degradation = self.last_soh_values[0] - self.last_soh_values[-1]
        total_episodes = len(self.last_soh_values) - 1
        degradation_per_episode = total_degradation / total_episodes
        degradation_per_hour = degradation_per_episode / hours_per_episode
        
        return degradation_per_hour
    
    def project_time_to_eol(self, degradation_rate):
        """Project time to reach end-of-life based on current degradation rate"""
        if degradation_rate <= 0 or not self.last_soh_values:
            return float('inf')
            
        current_soh = self.last_soh_values[-1]
        soh_remaining = current_soh - self.config.simulation.soh_min
        hours_remaining = soh_remaining / degradation_rate
        
        return hours_remaining

def make_env(path, journalist, rank, seed=0):
    """
    Create a function that will create and wrap the environment for multiprocessing
    """
    def _init():
        env = create_env(path, journalist_ins=journalist)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def setup_environment(path, journalist, num_envs=1, use_subproc=False):
    """
    Set up vectorized environment with option for subprocesses
    """
    env_fns = [make_env(path, journalist, i) for i in range(num_envs)]
    
    if use_subproc and num_envs > 1:
        # Use subprocess vectorization for truly parallel environments
        env = SubprocVecEnv(env_fns)
    else:
        # Use dummy vectorization (sequential execution)
        env = DummyVecEnv(env_fns)
    
    env = VecNormalize(
        env,
        norm_obs=False,      # Don't normalize observations since we want to retain physical meaning
        norm_reward=True,    # Normalize rewards for stable training
        clip_reward=10.0,    # Clip rewards to prevent extreme values
        gamma=0.99,
        epsilon=1e-08
    )
    return env

def create_sac_model(env, device='auto', **kwargs):
    """
    Create a SAC model with optimized hyperparameters for the BMS environment
    """
    # Determine observation and action dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Policy kwargs for network architecture
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],    # Policy network
            qf=[256, 256]     # Q-functions
        ),
        activation_fn=torch.nn.ReLU,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=dict(eps=1e-5)
    )
    
    # Create SAC model
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1000000,         # 1M transitions in replay buffer
        learning_starts=100,         # Collect this many transitions before training
        batch_size=256,              # Batch size for updating
        tau=0.005,                   # Soft update coefficient
        gamma=0.99,                  # Discount factor
        train_freq=1,                # Update policy every step
        gradient_steps=1,            # How many gradient steps after each rollout
        action_noise=None,           # SAC has built-in exploration, no need for extra noise
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log="Code/results/tensorboard_logs",
        ent_coef='auto',             # Automatic entropy coefficient tuning
        **kwargs
    )
    
    return model

def setup_callbacks(save_freq=10000, eval_freq=10000):
    """
    Set up training callbacks for checkpoints, evaluation, and logging
    """
    # Create directories if they don't exist
    os.makedirs("Code/results/checkpoints", exist_ok=True)
    os.makedirs("Code/results/best_model", exist_ok=True)
    
    # Checkpoint callback - saves model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="Code/results/checkpoints",
        name_prefix="sac_model"
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
                    eval_env = DummyVecEnv([make_env('src/configuration/config.yml', None, 0, seed=42)])
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

def train_and_evaluate(total_timesteps, num_envs=4, use_subproc=True, device="auto"):
    """
    Main function to train SAC model and evaluate it
    """
    try:
        # Create directories
        os.makedirs("Code/results/final_models", exist_ok=True)
        os.makedirs("Code/results/tensorboard_logs", exist_ok=True)
        os.makedirs("Code/results/training_data", exist_ok=True)
        
        # Get current time for unique naming
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')
        
        # Setup logging
        journalist = Journal("Code/results/Training", "sac_training_" + formatted_time)
        
        # Load config to get max_steps_per_episode
        config = load_config('src/configuration/config.yml')
        max_steps_per_episode = config.environment.max_steps_per_episode
        journalist._process_smoothly(f"Using max_steps_per_episode: {max_steps_per_episode}")
        
        # Setup environment - use multiple environments for faster training
        journalist._process_smoothly(f"Setting up {num_envs} environments with {'subprocesses' if use_subproc else 'sequential processing'}")
        env = setup_environment('src/configuration/config.yml', journalist, num_envs=num_envs, use_subproc=use_subproc)
        
        # Create callbacks
        tensorboard_callback = TensorboardCallback()
        checkpoint_callback = CheckpointCallback(
            save_freq=1000,  # Save every 1000 steps (divided by num_envs for actual steps)
            save_path=f"Code/results/checkpoints/SAC_{formatted_time}/",
            name_prefix="sac_model"
        )
        
        # Create reset callback to track episode resets due to battery waste or data completion
        reset_callback = ResetTrackingCallback(config=config)
        
        # Combine callbacks
        callbacks = CallbackList([tensorboard_callback, checkpoint_callback, reset_callback])
        
        # Create model
        journalist._process_smoothly(f"Creating SAC model on device: {device}")
        model = create_sac_model(env, device=device)
        
        journalist._process_smoothly("SAC model created successfully")
        journalist._process_smoothly(f"Starting training for {total_timesteps} timesteps")
        
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=f"SAC_{formatted_time}",
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
            step_df.to_csv(f"Code/results/training_data/SAC_steps_{formatted_time}.csv")
            episode_df.to_csv(f"Code/results/training_data/SAC_episodes_{formatted_time}.csv")
            
            # Plot training data
            journalist._plot_training_results(step_df, episode_df, formatted_time)
        
        # Calculate battery statistics
        time_interval = config.simulation.time_interval  # in hours
        total_sim_hours = model.num_timesteps * time_interval
        total_sim_years = total_sim_hours / (24 * 365)
        
        # Calculate expected battery replacements for 10 years
        avg_lifetime_hours = 0
        replacements_10years = 0
        if reset_callback.battery_eol_timestamps:
            avg_lifetime_hours = reset_callback.calculate_avg_lifetime()
            if avg_lifetime_hours > 0:
                replacements_10years = 10 * 365 * 24 / avg_lifetime_hours
        
        # Generate battery lifetime report
        battery_report = f"""
Battery Lifetime Report:
-----------------------
SOH minimum threshold: {config.simulation.soh_min * 100:.1f}%
Total simulation time: {total_sim_hours:.1f} hours ({total_sim_years:.2f} years)
Battery replacements: {reset_callback.resets_due_to_battery_waste}
Average battery lifetime: {avg_lifetime_hours:.1f} hours ({avg_lifetime_hours/(24*365):.2f} years)
Projected replacements per 10 years: {replacements_10years:.2f}
        """
        
        # Save battery report to file
        os.makedirs("Code/results/battery_reports", exist_ok=True)
        with open(f"Code/results/battery_reports/report_{formatted_time}.txt", "w") as f:
            f.write(battery_report)
        
        # Log training results
        journalist._process_smoothly(
            f"Training completed.\n"
            f"Final reward: {final_reward:.2f}\n"
            f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}\n"
            f"Battery waste resets: {reset_callback.resets_due_to_battery_waste}\n"
            f"Data completion resets: {reset_callback.resets_due_to_data_completion}\n"
            f"\n{battery_report}"
        )
        
        # Evaluate the model
        try:
            if num_envs > 1 and use_subproc:
                # Close the parallel environments
                env.close()
            
            # Create a single environment for evaluation
            eval_env = setup_environment('src/configuration/config.yml', journalist, num_envs=1)
            eval_rewards = []
            eval_episode_lengths = []
            eval_battery_wastes = 0
            n_eval_episodes = 5
            
            for i in range(n_eval_episodes):
                episode_reward = 0
                obs = eval_env.reset()[0]
                done = False
                step_count = 0
                episode_info = None
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = eval_env.step(action)
                    episode_reward += reward
                    step_count += 1
                    episode_info = info
                    done = done or truncated
                
                eval_rewards.append(episode_reward)
                eval_episode_lengths.append(step_count)
                
                # Check if episode ended due to battery waste
                final_soh = episode_info.get('soh', [1.0])[-1] if isinstance(episode_info.get('soh', 1.0), list) else episode_info.get('soh', 1.0)
                if final_soh <= config.simulation.soh_min:
                    eval_battery_wastes += 1
                
                journalist._process_smoothly(
                    f"Evaluation episode {i+1}: Reward = {episode_reward:.2f}, "
                    f"Steps = {step_count}, "
                    f"Ending SOH = {final_soh:.4f}"
                )
            
            mean_reward = np.mean(eval_rewards)
            std_reward = np.std(eval_rewards)
            mean_length = np.mean(eval_episode_lengths)
            
            journalist._process_smoothly(
                f"Evaluation complete.\n"
                f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}\n"
                f"Mean episode length: {mean_length:.1f} steps\n"
                f"Battery waste episodes: {eval_battery_wastes}/{n_eval_episodes}"
            )
            
        except Exception as e:
            journalist._get_error(f"Evaluation failed: {str(e)}")
            mean_reward, std_reward = 0.0, 0.0
        finally:
            # Make sure to close environments
            if 'eval_env' in locals():
                eval_env.close()
        
        # Save final model
        model.save(f"Code/results/final_models/SAC_final_{formatted_time}")
        journalist._process_smoothly(f"Model saved to Code/results/final_models/SAC_final_{formatted_time}")
        
        return model, env
        
    except Exception as e:
        journalist._get_error(f"Training failed: {str(e)}")
        raise e
    finally:
        # Make sure we close environments to avoid resource leaks
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    set_random_seed(42)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} for training")
    
    # Training parameters
    data_length = 34700
    num_passes = 1
    total_steps = data_length * num_passes
    
    # Number of parallel environments (set to 1 for debugging or 4-8 for faster training)
    num_envs = 4 if device == "cuda" else 2
    
    # Train model
    model, env = train_and_evaluate(
        total_timesteps=total_steps,
        num_envs=num_envs,
        use_subproc=True,  # Set to False for debugging
        device=device
    )
    
    # Close environment
    env.close()
    
    if device == "cuda":
        print(f"Final GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB") 