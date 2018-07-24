"""
Neyboy Challenge Learning Environment execution.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from copy import deepcopy
from pathlib import Path

import numpy as np
from tensorforce import TensorForceError
from tensorforce.agents import agents as AgentsDictionary, Agent
from tensorforce.execution import ThreadedRunner
from tensorforce.execution.threaded_runner import WorkerAgentGenerator

from neyboy.environment import NeyboyEnvironment

"""
python -m neyboy.threaded_neyboy --load checkpoints4/ --save checkpoints5/ -a configs/ppo.json -n configs/cnn.json -fs 2 -w 2
"""


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--agent-config', default='configs/ppo.json', help="Agent configuration file")
    parser.add_argument('-n', '--network-spec', default='configs/cnn.json', help="Network specification file")
    parser.add_argument('-w', '--workers', help="Number of threads to run where the model is shared", type=int,
                        default=2)
    parser.add_argument('-fs', '--frame-skip', help="Number of frames to repeat action", type=int, default=2)
    parser.add_argument('-ea', '--epsilon-annealing', help='Create separate epislon annealing schedules per thread',
                        action='store_true')
    parser.add_argument('-ds', '--display-screen', action='store_true', default=False, help="Display browser screen")
    parser.add_argument('-e', '--episodes', type=int, default=50000, help="Number of episodes")
    parser.add_argument('-t', '--max-timesteps', type=int, default=2000, help="Maximum number of timesteps per episode")
    parser.add_argument('-s', '--save', help="Save agent to this dir")
    parser.add_argument('-se', '--save-frequency', type=int, default=100, help="Save agent every x episodes")
    parser.add_argument('-l', '--load', help="Load agent from this dir")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # configurable!!!
    logger.addHandler(logging.StreamHandler(sys.stdout))

    environments = [NeyboyEnvironment(frame_skip=args.frame_skip,
                                      headless=not args.display_screen,
                                      user_data_dir='.tmp/puppetteer_{}/'.format(env_id)) for env_id in range(args.workers)]

    if args.network_spec:
        with open(args.network_spec, 'r') as fp:
            network_spec = json.load(fp=fp)
    else:
        network_spec = None
        logger.info("No network configuration provided.")

    agent_configs = []
    if args.agent_config is not None:
        with open(args.agent_config, 'r') as fp:
            agent_config = json.load(fp=fp)
    else:
        raise TensorForceError("No agent configuration provided.")

    for i in range(args.workers):
        worker_config = deepcopy(agent_config)
        # Optionally overwrite epsilon final values
        if "explorations_spec" in worker_config and worker_config['explorations_spec']['type'] == "epsilon_anneal":
            if args.epsilon_annealing:
                # epsilon final values are [0.5, 0.1, 0.01] with probabilities [0.3, 0.4, 0.3]
                epsilon_final = np.random.choice([0.5, 0.1, 0.01], p=[0.3, 0.4, 0.3])
                worker_config['explorations_spec']["epsilon_final"] = epsilon_final
        agent_configs.append(worker_config)

    # Let the first agent create the model
    # Manually assign model
    logger.info(agent_configs[0])

    agent = Agent.from_spec(
        spec=agent_configs[0],
        kwargs=dict(
            states=environments[0].states,
            actions=environments[0].actions,
            network=network_spec
        )
    )

    agents = [agent]

    for i in range(args.workers - 1):
        config = agent_configs[i]
        agent_type = config.pop('type', None)
        worker = WorkerAgentGenerator(AgentsDictionary[agent_type])(
            states=environments[0].states,
            actions=environments[0].actions,
            network=network_spec,
            model=agent.model,
            **config
        )
        agents.append(worker)

    if args.load:
        load_dir = Path(args.load)
        if not load_dir.is_dir():
            raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
        agent.restore_model(str(load_dir))

    if args.debug:
        logger.info("-" * 16)
        logger.info("Configuration:")
        logger.info(agent_configs[0])

    if args.save:
        save_dir = Path(args.save)
    else:
        save_dir = Path(str(uuid.uuid4()))

    def episode_finished(stats):
        if args.debug:
            logger.info(
                "Thread {t}. Finished episode {ep} after {ts} timesteps. Reward {r}".
                    format(t=stats['thread_id'], ep=stats['episode'], ts=stats['timestep'], r=stats['episode_reward'])
            )
        return True

    def summary_report(r):
        et = time.time()
        logger.info('=' * 40)
        logger.info('Current Step/Episode: {}/{}'.format(r.global_step, r.global_episode))
        logger.info('SPS: {}'.format(r.global_step / (et - r.start_time)))
        reward_list = r.episode_rewards
        if len(reward_list) > 0:
            logger.info('Max Reward: {}'.format(np.max(reward_list)))
            logger.info("Average of last 500 rewards: {}".format(sum(reward_list[-500:]) / 500))
            logger.info("Average of last 100 rewards: {}".format(sum(reward_list[-100:]) / 100))
        logger.info('=' * 40)

    # Create runners
    threaded_runner = ThreadedRunner(
        agents,
        environments,
        repeat_actions=args.frame_skip,
        save_path=str(save_dir),
        save_frequency=args.save_frequency
    )

    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environments[0]))
    threaded_runner.run(summary_interval=100, episode_finished=episode_finished, summary_report=summary_report)
    threaded_runner.close()
    logger.info("Learning finished. Total episodes: {ep}".format(ep=threaded_runner.global_episode))


if __name__ == '__main__':
    main()
