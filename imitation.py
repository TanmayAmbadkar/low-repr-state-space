from env import CustomBipedalWalker
from ppo.policy import train_policy
from stable_baselines3 import PPO
from sb3_utils import learn
import sys
from sklearn.decomposition import PCA
import numpy as np
import statistics
from transforms import Autoencoder, fit
import torch
# Create the BipedalWalker environment
env = CustomBipedalWalker()


# Create the PPO agent
policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                     net_arch=dict(pi=[32, 8], vf=[32, 8]))
agent = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs,clip_range=0.18, gae_lambda=0.95, gamma = 0.999, learning_rate=0.0003)

# Train the agent
_, observations = learn(agent, total_timesteps=5000000)
print()
observations = np.array(observations)
observations = observations.reshape(observations.shape[0], -1)  
# umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=4)

# model = Autoencoder(observations.shape[1], 4)
# fit(observations=observations, autoencoder=model)

# model = PCA(n_components=4)
# model.fit(observations)

print("Fitted AE model, transforming the environment")

env_reduced = CustomBipedalWalker(agent.policy.mlp_extractor.policy_net, reduced_dim=8)
agent_reduced = PPO("MlpPolicy", env_reduced, verbose=1)
_, _ = learn(agent_reduced, total_timesteps=5000000)
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