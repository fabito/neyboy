import argparse
import logging
import os
import time

from PIL import Image
from neyboy_tensorforce.agents import PPOAgent
from neyboy_tensorforce.execution import Runner

from neyboy_tensorforce.runner import MetricsLoggingRunner
from .environment import NeyboyEnvironment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--episodes', type=int, default=30000, help="Number of episodes")
    parser.add_argument('-t', '--timesteps', type=int, default=None, help="Number of timesteps")
    parser.add_argument('-b', '--batch-size', type=int, default=64, help="Batch Size")
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-5, help="Learning Rate")
    parser.add_argument('-st', '--score-threshold', type=float, default=0.95, help="Score threshold")
    parser.add_argument('-m', '--max-episode-timesteps', type=int, default=2000,
                        help="Maximum number of timesteps per episode")
    parser.add_argument('-r', '--repeat-actions', type=int, default=4,
                        help="repeat_actions")
    parser.add_argument('-d', '--deterministic', action='store_true', default=False,
                        help="Choose actions deterministically")
    parser.add_argument('-s', '--save-dir', help="Save agent to this dir")
    parser.add_argument('-se', '--save-episodes', type=int, default=100, help="Save agent every x episodes")
    parser.add_argument('-l', '--load', help="Load agent from this dir")
    parser.add_argument('--visualize', action='store_true', default=False, help="Enable OpenAI Gym's visualization")
    parser.add_argument('-D', '--debug', action='store_true', default=True, help="Show debug outputs")
    parser.add_argument('-te', '--test', action='store_true', default=False, help="Test agent without learning.")
    parser.add_argument('-sl', '--sleep', type=float, default=None,
                        help="Slow down simulation by sleeping for x seconds (fractions allowed).")
    parser.add_argument('-R', '--random-test-run', action="store_true", help="Do a quick random test run on the env")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    environment = NeyboyEnvironment(headless=not args.visualize,
                                    score_threshold=args.score_threshold)

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
            batch_size=args.batch_size,
            # Every 10 episodes
            frequency=10
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
            learning_rate=args.learning_rate
        ),

        # OOM
        # baseline_mode="states",
        # baseline={
        #     "type": "cnn",
        #     "conv_sizes": [32, 32],
        #     "dense_sizes": [32]
        # },
        # baseline_optimizer={
        #     "type": "multi_step",
        #     "optimizer": {
        #         "type": "adam",
        #         "learning_rate": 1e-3
        #     },
        #     "num_steps": 5
        # },

        # saver=dict(
        #     directory=args.save_dir,
        #     seconds=600
        # ),

        summarizer=dict(
            directory=args.save_dir,
            labels=['rewards'],
            steps=50
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

    runner = MetricsLoggingRunner(
        agent=agent,
        environment=environment,
        repeat_actions=args.repeat_actions,
        log_dir=args.save_dir
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

        if args.save_dir and args.save_episodes is not None and not r.episode % args.save_episodes:
            logger.info("Saving agent to {}".format(args.save_dir))
            r.agent.save_model(args.save_dir)

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
