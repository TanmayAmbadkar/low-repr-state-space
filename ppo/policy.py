import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from tqdm import trange
import statistics


from ppo.utils import RolloutBuffer
from ppo.actor_critic import ActorCritic
class PPO:
    def __init__(self, state_dim, action_dim, continuous, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        
        self.buffer = RolloutBuffer(batch_size=64)

        self.policy = ActorCritic(state_dim, action_dim, continuous).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic},
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, continuous).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())


    def __call__(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.cpu().numpy() 


    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)


    def update(self):
        # Prepare data for training
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Get data in batches
            for batch_index in range(0, len(old_states), self.buffer.batch_size):
                batch_states = old_states[batch_index:batch_index + self.buffer.batch_size]
                batch_actions = old_actions[batch_index:batch_index + self.buffer.batch_size]
                batch_old_logprobs = old_logprobs[batch_index:batch_index + self.buffer.batch_size]
                batch_rewards = rewards[batch_index:batch_index + self.buffer.batch_size]

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(batch_states, batch_actions)

                # Match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta_old)
                ratios = torch.exp(logprobs - batch_old_logprobs.detach())

                # Finding Surrogate Loss
                advantages = batch_rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # Final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(state_values, batch_rewards) - 0.01 * dist_entropy

                # Take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()

        
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


def sample_policy(env: gym.Env, policy:PPO):
    
    observation, _ = env.reset()
    total_reward = 0
    while True:

        action = policy(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward+=reward

        policy.buffer.rewards.append(reward)
        
        policy.buffer.is_terminals.append(terminated)

        if terminated or truncated:
            break
    
    return total_reward



def train_policy(env: gym.Env, n_episodes=3000, minimum_reach=0.9):

    policy = PPO(
        state_dim = env.observation_space.shape[0],
        action_dim = env.action_space.shape[0],
        continuous=True,
        lr_actor=0.0003,
        lr_critic=0.0003,
        gamma = 0.9,
        K_epochs = 40,
        eps_clip = 0.2,
        device = "cpu"
    )


    rewards = [0]
    episodes = trange(n_episodes, desc='reach')
    for episode in episodes:
        
        
        reward = sample_policy(env, policy)
        rewards.append(reward)

        episodes.set_description(f"Reward: {statistics.mean(rewards):.2f}±{statistics.stdev(rewards):.1f}")
       
        if episode % 10 == 0:
            policy.update()
    

    return policy


