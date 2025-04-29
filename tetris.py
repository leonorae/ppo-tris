import sys
from copy import deepcopy

import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

# tetris_gymnasium is not defined in any of the internal baseline packages
# and need deepcopy later
tetris_env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array")

# class HeuristicReward(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.obs = None
        
#     def observation(self, obs):
#         self.obs = obs

#         return obs

#     def reward(self, def_r):
#         # - The height of the stack in each column (list: int for each column)
#         # - The maximum height of the stack (int)
#         # - The number of holes in the stack (int)
#         # - The bumpiness of the stack (int)

#         new_r = def_r - ( obs[3] * 0.3 + obs[2] * 0.4 + sum(obs[0]) * 0.5)

#         return new_r

def make_env(rank: int, seed: int = 0):
    """
    make a new environment instance for parallel processing and wrap the environment (as a thunk to be called interally by baselines)
    """

    def wrapped_env():
        env = deepcopy(tetris_env)
        env = GroupedActionsObservations(env, observation_wrappers=[FeatureVectorObservation(env)])
        # env = HeuristicReward(env)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    
    return wrapped_env

wrapped_env = make_env(1, 335)
# i have an 8 core cpu, so use 8 parallel envs to train
vec_env = make_vec_env(wrapped_env, n_envs=8, vec_env_cls=SubprocVecEnv)
vec_env = VecNormalize(vec_env)
model = PPO("MlpPolicy",
            vec_env,
            verbose=1,
            clip_range=0.2,
            learning_rate=0.0001,
            gamma=0.99,
            device="cpu")

model.learn(total_timesteps=100_000_000)

model.save("ppo_tetris_grouped_feature_hyperparam_100M")

# model = PPO.load("ppo_tetris_grouped_feature_heuristicadj2_hyperparam_5M")

obs = vec_env.reset()
# mean_reward, std_reward = evaluate_policy(model, obs, n_eval_episodes=1_000)
# print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")


# new env for human display and play back with 4 because it looks nicer
playback = make_vec_env(wrapped_env,  n_envs = 4)
playback.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = playback.step(action)
    playback.render("human")
        
# TODO: Make RandomAgent
# if __name__ == "__main__":
#     env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
#     env.reset(seed=42)

#     episode_over = False
#     while not episode_over:
#         env.render()
#         action = env.action_space.sample()
#         env.has_wrapper_attr
#observation, reward, terminated, truncated, info = env.step(action)
#         key = cv2.waitKey(100) # timeout to see the movement

#         episode_over = terminated or truncated

#     print("Game Over!")


