import argparse
import json
import logging
import os
import time

from PIL import Image
from tensorforce import TensorForceError
from tensorforce.agents import Agent
from tensorforce.execution import Runner

from .environment import NeyboyEnvironment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--agent', help="Agent configuration file")
    parser.add_argument('-n', '--network', default=None, help="Network specification file")
    parser.add_argument('-e', '--episodes', type=int, default=30000, help="Number of episodes")
    parser.add_argument('-t', '--timesteps', type=int, default=None, help="Number of timesteps")
    parser.add_argument('-m', '--max-episode-timesteps', type=int, default=2000,
                        help="Maximum number of timesteps per episode")
    parser.add_argument('-r', '--repeat-actions', type=int, default=4,
                        help="repeat_actions")
    parser.add_argument('-d', '--deterministic', action='store_true', default=False,
                        help="Choose actions deterministically")
    parser.add_argument('-s', '--save', help="Save agent to this dir")
    parser.add_argument('-se', '--save-episodes', type=int, default=100, help="Save agent every x episodes")
    parser.add_argument('-l', '--load', help="Load agent from this dir")
    parser.add_argument('--monitor', help="Save results to this directory")
    parser.add_argument('--monitor-safe', action='store_true', default=False, help="Do not overwrite previous results")
    parser.add_argument('--monitor-video', type=int, default=0, help="Save video every x steps (0 = disabled)")
    parser.add_argument('--visualize', action='store_true', default=False, help="Enable OpenAI Gym's visualization")
    parser.add_argument('-D', '--debug', action='store_true', default=True, help="Show debug outputs")
    parser.add_argument('-te', '--test', action='store_true', default=False, help="Test agent without learning.")
    parser.add_argument('-sl', '--sleep', type=float, default=None,
                        help="Slow down simulation by sleeping for x seconds (fractions allowed).")
    parser.add_argument('-R', '--random-test-run', action="store_true", help="Do a quick random test run on the env")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    if args.save:
        save_dir = os.path.dirname(args.save)
        if not os.path.isdir(save_dir):
            try:
                os.mkdir(save_dir, 0o755)
            except OSError:
                raise OSError("Cannot save agent to dir {} ()".format(save_dir))

    environment = NeyboyEnvironment(headless=not args.visualize, frame_skip=2.5)

    # Do a quick random test-run with image capture of the first n images -> then exit after 1000 steps.
    if args.random_test_run:
        # Reset the env.
        s = environment.reset()
        img_format = "RGB" if len(environment.states["shape"]) == 3 else "L"
        img = Image.fromarray(s, img_format)
        # Save first received image as a sanity-check.
        img.save("reset.jpg")
        step = 0
        for i in '1111011111100000111011110110001111100001111100000011100111000111100100101100011100101110010110000111100011101011011001101110101111000001':
            action = int(i) + 1
            s, is_terminal, r = environment.execute(actions=action)
            img = Image.fromarray(s, img_format)
            img.save("{:03d}.jpg".format(step))
            logging.debug("i={} r={} term={}".format(step, r, is_terminal))
            step += 1
            if is_terminal:
                break
        # input("Press Enter to continue...")
        environment.close()
        quit()

    if args.network_spec:
        with open(args.network_spec, 'r') as fp:
            network_spec = json.load(fp=fp)
    else:
        network_spec = None
        logger.info("No network configuration provided.")

    if args.agent_config is not None:
        with open(args.agent_config, 'r') as fp:
            agent_config = json.load(fp=fp)
    else:
        raise TensorForceError("No agent configuration provided.")

    agent = Agent.from_spec(
        spec=agent_config,
        kwargs=dict(
            states=environment.states,
            actions=environment.actions,
            network=network_spec
        )
    )

    if args.load:
        load_dir = os.path.dirname(args.load)
        if not os.path.isdir(load_dir):
            raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
        agent.restore_model(args.load)

    if args.debug:
        logger.info("-" * 16)
        logger.info("Configuration:")
        logger.info(agent)

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=args.repeat_actions
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
        num_episodes=args.episodes,
        max_episode_timesteps=args.max_episode_timesteps,
        deterministic=args.deterministic,
        episode_finished=episode_finished,
    )
    runner.close()

    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))


if __name__ == '__main__':
    main()
