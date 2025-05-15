import numpy as np
from gymnasium import Wrapper

class RewardSumFixWrapper(Wrapper):
    """
    A wrapper to fix the 'list' object has no attribute 'sum' error.
    This converts list rewards to numpy arrays when needed.
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
    def step(self, action):
        try:
            # Try the normal step method
            obs, reward, terminated, truncated, info = self.env.step(action)
            return obs, reward, terminated, truncated, info
        except AttributeError as e:
            if "'list' object has no attribute 'sum'" in str(e):
                # Fix the issue by converting the reward list to a numpy array
                if hasattr(self.env, 'info') and 'reward' in self.env.info and isinstance(self.env.info['reward'], list):
                    self.env.info['reward'] = np.array(self.env.info['reward'])
                    # Try the step again
                    return self.env.step(action)
            # If it's a different error, re-raise it
            raise e 