import numpy as np

from tensorforce.agents import DQNAgent, PPOAgent, RandomAgent
from tensorforce.execution import Runner

from model import NeyboyEnvironment

environment = NeyboyEnvironment(headless=True, frame_skip=1)

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
        batch_size=10,
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
    # }
    saver={
         "directory": 'saved',
         "seconds": 600
    }

)

# Create the runner
runner = Runner(agent=agent, environment=environment)


# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    return True


# Start learning
runner.run(episodes=30000, max_episode_timesteps=2000, episode_finished=episode_finished)
runner.close()

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)





