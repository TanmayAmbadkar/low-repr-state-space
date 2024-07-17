from env.env import CustomBipedalWalker
from stable_baselines3 import PPO
from sb3_utils import learn
import sys
from sklearn.decomposition import PCA
import numpy as np
import statistics
from transforms import Autoencoder, fit
import torch
from stable_baselines3.common.utils import set_random_seed
from abstract_interpretation.domains import Zonotope

def create_environment(state_processor = None, reduced_dim = None, seed=10):
    """
    Create the CustomBipedalWalker environment with a specified random seed.
    
    Parameters:
    seed (int): The random seed for reproducibility.
    
    Returns:
    env (CustomBipedalWalker): The initialized environment.
    """
    set_random_seed(seed)
    return CustomBipedalWalker(state_processor, reduced_dim)

def create_agent(env, policy_kwargs=None, device="cpu"):
    """
    Create the PPO agent for the specified environment.
    
    Parameters:
    env (CustomBipedalWalker): The environment for the agent.
    policy_kwargs (dict): The policy keyword arguments for the PPO agent.
    device (str): The device to run the agent on ("cpu" or "cuda").
    
    Returns:
    agent (PPO): The initialized PPO agent.
    """
    return PPO("MlpPolicy", env, verbose=1, device=device, policy_kwargs=policy_kwargs)

def train_agent(agent, total_timesteps=100000):
    """
    Train the PPO agent for the specified number of timesteps.
    
    Parameters:
    agent (PPO): The PPO agent to be trained.
    total_timesteps (int): The number of timesteps to train the agent.
    
    Returns:
    agent (PPO): The trained PPO agent.
    observations (list): The list of observations collected during training.
    """
    _, observations = learn(agent, total_timesteps=total_timesteps)
    return agent, np.array(observations)

def fit_autoencoder(observations, input_dim, reduced_dim=3):
    """
    Fit an Autoencoder model to the observations.
    
    Parameters:
    observations (ndarray): The observations to fit the Autoencoder model.
    input_dim (int): The input dimension of the Autoencoder model.
    reduced_dim (int): The reduced dimension for the Autoencoder model.
    
    Returns:
    model (Autoencoder): The fitted Autoencoder model.
    """
    model = Autoencoder(input_dim, reduced_dim)
    fit(observations=observations, autoencoder=model)
    return model

def evaluate_agent(env, agent, episodes=5):
    """
    Evaluate the PPO agent in the specified environment.
    
    Parameters:
    env (CustomBipedalWalker): The environment to evaluate the agent.
    agent (PPO): The PPO agent to be evaluated.
    episodes (int): The number of episodes to evaluate the agent.
    
    Returns:
    rewards (list): The list of rewards for each episode.
    episode_lengths (list): The list of episode lengths for each episode.
    """
    rewards = []
    episode_lengths = []
    for _ in range(episodes):
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
            length += 1
        rewards.append(reward_ep)
        episode_lengths.append(length)
    return rewards, episode_lengths


def verify_network(model, input_domain):
    abstract_element = input_domain
    with torch.no_grad():
        for layer in model.layers:
            abstract_element = layer(abstract_element)
    
    return abstract_element

def main():
    # Create the BipedalWalker environment
    env = create_environment()

    # Create the PPO agent
    # policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=dict(pi=[16, 4], vf=[16, 4]))
    agent = create_agent(env)

    # Train the agent
    agent, observations = train_agent(agent)
    observations = observations.reshape(observations.shape[0], -1)

    # Fit the Autoencoder model
    # Change autoencoder to desired model, create function above
    model = fit_autoencoder(observations, input_dim=observations.shape[1])
    
    
    # Define the observation space bounds
    obs_space_lower = np.array([-1.5, -1.5, -5.0, -5.0, -3.1415927, -5.0, 0.0, 0.0])
    obs_space_upper = np.array([ 1.5,  1.5,  5.0,  5.0,  3.1415927,  5.0, 1.0, 1.0])

    # Calculate the center of the zonotope
    center = (obs_space_lower + obs_space_upper) / 2

    # Create generators to reflect the constraints on vx and vy
    generators = []

    # vx and vy constraints (-2 <= vx, vy <= 2)
    vx_gen = np.zeros(8)
    vx_gen[2] = 2

    vy_gen = np.zeros(8)
    vy_gen[3] = 2

    # Add these generators to the list
    generators.append(vx_gen)
    generators.append(vy_gen)

    # Additional generators for other dimensions can be added if necessary
    # Example: small perturbations in other dimensions
    for i in range(8):
        if i not in [2, 3]:
            gen = np.zeros(8)
            gen[i] = (obs_space_upper[i] - obs_space_lower[i]) / 2
            generators.append(gen)

    # Create the zonotope
    input_zonotope = Zonotope(center, generators)

    # Print the zonotope details
    hyperplanes = input_zonotope.to_hyperplanes()
    print("Initial Zonotope details")
    for A, b in hyperplanes:
        print(f"Hyperplane: {A} * x <= {b}")
    
    final_zonotope = verify_network(model.encoder, input_zonotope)
    
    print("Final Zonotope details")
    hyperplanes = final_zonotope.to_hyperplanes()
    for A, b in hyperplanes:
        print(f"Hyperplane: {A} * x <= {b}")
    
        
    # # Create the reduced environment and agent
    # env_reduced = create_environment(model.encoder, reduced_dim=3)
    
    # agent_reduced = create_agent(env_reduced)

    # # Train the reduced agent
    # agent_reduced, _ = train_agent(agent_reduced)

    # # Evaluate the original agent
    # rewards, episode_lengths = evaluate_agent(env, agent)

    # # Evaluate the reduced agent
    # rewards_reduced, episode_lengths_reduced = evaluate_agent(env_reduced, agent_reduced)

    # # Print evaluation results
    # print(f"Original state space reward: {statistics.mean(rewards)}+-{statistics.stdev(rewards)}")
    # print(f"Original state space length: {statistics.mean(episode_lengths)}+-{statistics.stdev(episode_lengths)}")
    # print(f"Modified state space reward: {statistics.mean(rewards_reduced)}+-{statistics.stdev(rewards_reduced)}")
    # print(f"Modified state space length: {statistics.mean(episode_lengths_reduced)}+-{statistics.stdev(episode_lengths_reduced)}")

if __name__ == "__main__":
    main()
