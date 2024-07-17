import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data.types import TrajectoryWithRew
from env import CustomBipedalWalker
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from sb3_utils import learn
import statistics
from transforms import Autoencoder, fit
import torch


# env = CustomBipedalWalker(state_processor=None, reduced_dim = None)
env = gym.make("LunarLander-v2")

rng = np.random.default_rng(0)
env = make_vec_env(
    "LunarLander-v2",
    rng=rng,
    n_envs=1,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
)
expert = PPO("MlpPolicy", env, verbose=1, device = "cpu")

# Train the agent
_, observations = learn(expert, total_timesteps=20000)
print()
observations = np.array(observations)
observations = observations.reshape(observations.shape[0], -1)  
# umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=4)

model = Autoencoder(observations.shape[1], 3)
fit(observations=observations, autoencoder=model)

# model = PCA(n_components=4)
# model.fit(observations)

print("Fitted AE model, transforming environment")

rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_timesteps=None, min_episodes=50),
    rng=rng,
)

with torch.no_grad():
    rollouts = TrajectoryWithRew(model.encoder(torch.Tensor(rollouts.obs)).numpy(), rollouts.sacts, rollouts.infos, rollouts.terminal,rollouts.rews)

transitions = rollout.flatten_trajectories(rollouts)

bc_trainer = bc.BC(
    observation_space=(3,),
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)
bc_trainer.train(n_epochs=10)
reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
print("Reward:", reward)