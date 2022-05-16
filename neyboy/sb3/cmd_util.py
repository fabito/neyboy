import argparse
import os

import gym
from gym import wrappers

import gym_neyboy
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, WarpFrame
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


def make_neyboy_environment(env_id: str, logdir: str, seed=0, rank=0, allow_early_resets=False, frame_skip=4,
                            save_video=False):
    env = gym.make(env_id)
    env = MaxAndSkipEnv(env, skip=frame_skip)
    env.seed(seed + rank)

    if save_video:
        env = wrappers.Monitor(env, logdir, force=True)
    else:
        env = Monitor(env, logdir, allow_early_resets=allow_early_resets)
    return env


def make_neyboy_env(env_id: str, num_env: int, logger: Logger, seed: int, start_index=0, allow_early_resets=False,
                    frame_skip=4, save_video=False):
    """
    Create a wrapped, monitored SubprocVecEnv for Neyboy.
    """
    def make_env(rank):
        logdir = logger.get_dir() and os.path.join(logger.get_dir(), str(rank))

        def _thunk() -> gym.Env:
            env = make_neyboy_environment(env_id, logdir, seed, rank, allow_early_resets, frame_skip=frame_skip,
                                          save_video=save_video)
            env = WarpFrame(env)
            return env
        return _thunk

    set_random_seed(seed)

    envs = [make_env(i + start_index) for i in range(num_env)]
    if num_env > 1:
        env = SubprocVecEnv(envs)
    else:
        env = DummyVecEnv(envs)

    return env


def neyboy_arg_parser():
    """
    Create an argparse.ArgumentParser for run_neyboy.py.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', default='neyboy-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))

    return parser
