import gymnasium as gym
import torch

class CustomBipedalWalker(gym.Env):
    def __init__(self, state_processor = None, reduced_dim = None):
        self.env = gym.make("LunarLander-v2")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space if state_processor is None else gym.spaces.Box(low=-1, high=1, shape=(reduced_dim,))
        self.state_processor = state_processor

    def step(self, action):
        
        state, reward, done, trunc, info = self.env.step(action)
        if self.state_processor is not None:
            state = torch.Tensor(state)
            with torch.no_grad():
                state = self.state_processor(state.reshape(1, -1))
            state = state.numpy()
        return state, reward, done, trunc, info

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        if self.state_processor is not None:
            state = torch.Tensor(state)
            with torch.no_grad():
                state = self.state_processor(state.reshape(1, -1))
            state = state.numpy()
        return state, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)