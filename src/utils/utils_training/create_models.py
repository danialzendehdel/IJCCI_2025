from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
import torch
import numpy as np

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
            tensorboard_log="results/tensorboard_logs",
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
            tensorboard_log="results/tensorboard_logs",
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
            tensorboard_log="results/tensorboard_logs",
            **kwargs
        )
    return model
