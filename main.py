import argparse
import logging
import os
import time

import numpy as np

from tensorforce.agents import DQNAgent, PPOAgent, RandomAgent
from tensorforce.execution import Runner

from environment import NeyboyEnvironment


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-a', '--agent', help="Agent configuration file")
    # parser.add_argument('-n', '--network', default=None, help="Network specification file")
    parser.add_argument('-e', '--episodes', type=int, default=30000, help="Number of episodes")
    parser.add_argument('-t', '--timesteps', type=int, default=None, help="Number of timesteps")
    parser.add_argument('-m', '--max-episode-timesteps', type=int, default=2000, help="Maximum number of timesteps per episode")
    parser.add_argument('-d', '--deterministic', action='store_true', default=False, help="Choose actions deterministically")
    parser.add_argument('-s', '--save', help="Save agent to this dir")
    parser.add_argument('-se', '--save-episodes', type=int, default=100, help="Save agent every x episodes")
    parser.add_argument('-l', '--load', help="Load agent from this dir")
    parser.add_argument('--monitor', help="Save results to this directory")
    parser.add_argument('--monitor-safe', action='store_true', default=False, help="Do not overwrite previous results")
    parser.add_argument('--monitor-video', type=int, default=0, help="Save video every x steps (0 = disabled)")
    parser.add_argument('--visualize', action='store_true', default=False, help="Enable OpenAI Gym's visualization")
    parser.add_argument('-D', '--debug', action='store_true', default=True, help="Show debug outputs")
    parser.add_argument('-te', '--test', action='store_true', default=False, help="Test agent without learning.")
    parser.add_argument('-sl', '--sleep', type=float, default=None, help="Slow down simulation by sleeping for x seconds (fractions allowed).")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    environment = NeyboyEnvironment(headless=not args.visualize, frame_skip=2.5)

    network_spec = [
        dict(type='conv2d', size=32, window=8, stride=4),
        dict(type='conv2d', size=64, window=4, stride=2),
        dict(type='conv2d', size=64, window=3, stride=1),
        dict(type='flatten'),
        dict(type='dense', size=512),
    ]

    preprocessing_config = [
        {
            "type": "image_resize",
            "width": 80,
            "height": 80
        },
        {
            "type": "grayscale"
        },
        {
            "type": "normalize"
        },
        {
            "type": "sequence",
            "length": 4
        }
    ]

    # agent = RandomAgent(
    #     states=environment.states,
    #     actions=environment.actions,
    # )

    # agent = DQNAgent()

    agent = PPOAgent(
        states=environment.states,
        actions=environment.actions,
        network=network_spec,
        # Agent
        states_preprocessing=preprocessing_config,
        actions_exploration=None,
        reward_preprocessing=None,
        # MemoryModel
        update_mode=dict(
            unit='episodes',
            # 10 episodes per update
            batch_size=32,
            # Every 10 episodes
            frequency=4
        ),
        memory=dict(
            type='latest',
            include_next_states=False,
            capacity=5000
        ),
        # DistributionModel
        distributions=None,
        entropy_regularization=0.01,
        execution=dict(
            type='single',
            session_config=None,
            distributed_spec=None
        ),
        step_optimizer=dict(
            type='adam',
            learning_rate=1e-5
        ),
        saver={
            "directory": 'saved',
            "seconds": 600
        }

    )

    if args.load:
        load_dir = os.path.dirname(args.load)
        if not os.path.isdir(load_dir):
            raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
        agent.restore_model(args.load)

    if args.save:
        save_dir = os.path.dirname(args.save)
        if not os.path.isdir(save_dir):
            try:
                os.mkdir(save_dir, 0o755)
            except OSError:
                raise OSError("Cannot save agent to dir {} ()".format(save_dir))

    if args.debug:
        logger.info("-" * 16)
        logger.info("Configuration:")
        logger.info(agent)

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )

    if args.debug:  # TODO: Timestep-based reporting
        report_episodes = 1
    else:
        report_episodes = 100

    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))

    def episode_finished(r, id_):
        if r.episode % report_episodes == 0:
            steps_per_second = r.timestep / (time.time() - r.start_time)
            logger.info("Finished episode {:d} after {:d} timesteps. Steps Per Second {:0.2f}".format(
                r.agent.episode, r.episode_timestep, steps_per_second
            ))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 500 rewards: {:0.2f}".
                        format(sum(r.episode_rewards[-500:]) / min(500, len(r.episode_rewards))))
            logger.info("Average of last 100 rewards: {:0.2f}".
                        format(sum(r.episode_rewards[-100:]) / min(100, len(r.episode_rewards))))
        if args.save and args.save_episodes is not None and not r.episode % args.save_episodes:
            logger.info("Saving agent to {}".format(args.save))
            r.agent.save_model(args.save)

        return True

    runner.run(
        # num_timesteps=args.timesteps,
        num_episodes=args.episodes,
        max_episode_timesteps=args.max_episode_timesteps,
        deterministic=args.deterministic,
        episode_finished=episode_finished,
        # testing=args.test,
        # sleep=args.sleep
    )
    runner.close()

    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))


if __name__ == '__main__':
    main()
