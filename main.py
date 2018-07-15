import numpy as np

from tensorforce.agents import DQNAgent
from tensorforce.execution import Runner
from tensorforce.core.preprocessors import Preprocessor

from model import NeyboyEnvironment

environment = NeyboyEnvironment(headless=False)

# model = Sequential()
# model.add(Conv2D(32, (8, 8), padding='same', strides=(4, 4), input_shape=(img_cols, img_rows, img_channels)))  # 80*80*4
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Activation('relu'))
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dense(ACTIONS))
# adam = Adam(lr=LEARNING_RATE)

network_spec = [
    dict(type='conv2d', size=32, window=8, stride=4),
    dict(type='conv2d', size=64, window=4, stride=2),
    dict(type='conv2d', size=64, window=3, stride=1),
    dict(type='flatten'),
    dict(type='dense', size=512),
    # dict(type='dense', size=32),
    # dict(type='lstm', size=32)
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
    }, {
        "type": "sequence",
        "length": 4
    }
]


agent = DQNAgent(
    states=environment.states,
    actions=environment.actions,
    network=network_spec,
    # Agent
    states_preprocessing=preprocessing_config,
    actions_exploration=None,
    reward_preprocessing=None,
    # MemoryModel
    update_mode=dict(
        unit='timesteps',
        # 10 episodes per update
        batch_size=100,
        # Every 10 episodes
        frequency=20
    ),
    memory=dict(
        type='latest',
        include_next_states=True,
        capacity=5000
    ),
    # DistributionModel
    distributions=None,
    entropy_regularization=0.01,
    execution=dict(
        type='single',
        session_config=None,
        distributed_spec=None
    )
)

# Create the runner
runner = Runner(agent=agent, environment=environment)


# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    return True


# Start learning
runner.run(episodes=3000, max_episode_timesteps=200, episode_finished=episode_finished)
runner.close()

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)





