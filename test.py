from env import CustomBipedalWalker
from stable_baselines3 import PPO
from sb3_utils import learn
import sys
from sklearn.decomposition import PCA
import numpy as np
import statistics
from transforms import Autoencoder, fit, CascadingAutoEncoder, fit_cascading_autoencoder
import torch
# Create the BipedalWalker environment
from stable_baselines3.common.utils import set_random_seed

set_random_seed(10)
env = CustomBipedalWalker()


# Create the PPO agent
policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                     net_arch=dict(pi=[16, 4], vf=[16, 4]))
agent = PPO("MlpPolicy", env, verbose=1, device = "cpu")

# Train the agent
_, observations = learn(agent, total_timesteps=500000)
print()
observations = np.array(observations)
observations = observations.reshape(observations.shape[0], -1)  
# umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=4)

# model = Autoencoder(observations.shape[1], 3)
# fit(observations=observations, autoencoder=model)

model = CascadingAutoEncoder(observations.shape[1], [32,16, 8, 4])
fit_cascading_autoencoder(observations, model)

# model = PCA(n_components=4)
# model.fit(observations)

print("Fitted (c)AE model, transforming the environment")

# env_reduced = CustomBipedalWalker(model.encoder, reduced_dim=3) # Replace model.encoder with different dimensionality reduction techniques
env_reduced = CustomBipedalWalker(model.encoder, reduced_dim=4)

agent_reduced = PPO("MlpPolicy", env_reduced, verbose=1, device = "cpu")
_, _ = learn(agent_reduced, total_timesteps=500000)
print()


# Test and render the agent for 5 episodes
rewards = []
episode_lengths = []
for _ in range(5):
    obs, _ = env.reset()
    done = False
    trunc = False
    reward_ep = 0
    length = 0
    while not (done or trunc):
        action, _ = agent.predict(obs)
        try:
            obs, reward, done, trunc, _ = env.step(action)
        except:
            print(action)
        reward_ep += reward
        length +=1
        
    rewards.append(reward_ep)
    episode_lengths.append(length)
    
env.close()

rewards_reduced = []
episode_lengths_reduced = []
for _ in range(5):
    obs, _ = env_reduced.reset()
    done = False
    trunc = False
    reward_ep = 0
    length = 0
    while not (done or trunc):
        action, _ = agent_reduced.predict(obs)
        try:
            obs, reward, done, trunc, _ = env_reduced.step(action[0])
        except Exception as e:
            print(e)
            print(action)
        reward_ep += reward
        length +=1
        
    rewards_reduced.append(reward_ep)
    episode_lengths_reduced.append(length)
    
print(f"Original state space reward: {statistics.mean(rewards)}+-{statistics.stdev(rewards)}")
print(f"Original state space length: {statistics.mean(episode_lengths)}+-{statistics.stdev(episode_lengths)}")
print(f"Modified state space reward: {statistics.mean(rewards_reduced)}+-{statistics.stdev(rewards_reduced)}")
print(f"Modified state space length: {statistics.mean(episode_lengths_reduced)}+-{statistics.stdev(episode_lengths_reduced)}")
