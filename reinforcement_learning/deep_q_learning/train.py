#!/usr/bin/env python3
"""Train a DQN agent to play Atari Breakout."""

import gymnasium as gym
import numpy as np

from keras.layers import Conv2D, Dense, Flatten, Permute
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy

try:
    import ale_py
    gym.register_envs(ale_py)
except (ImportError, AttributeError):
    pass


WINDOW_LENGTH = 4


class KerasRLWrapper(gym.Wrapper):
    """Make Gymnasium compatible with keras-rl2."""

    def reset(self, **kwargs):
        """Reset the environment using the old Gym API."""
        result = self.env.reset(**kwargs)

        if isinstance(result, tuple):
            return result[0]

        return result

    def step(self, action):
        """Perform an action using the old Gym API."""
        result = self.env.step(action)

        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            done = terminated or truncated

            return observation, reward, done, info

        return result

    def render(self, mode='human'):
        """Render the environment."""
        return self.env.render()


class AtariProcessor(Processor):
    """Preprocess Atari observations and rewards."""

    def process_state_batch(self, batch):
        """Normalize image pixel values."""
        batch = np.asarray(batch, dtype=np.float32)

        return batch / 255.0

    def process_reward(self, reward):
        """Clip rewards to the range [-1, 1]."""
        return np.clip(reward, -1.0, 1.0)


def make_env(render_mode='rgb_array'):
    """Create and preprocess the Breakout environment."""
    env = gym.make(
        'ALE/Breakout-v5',
        frameskip=1,
        repeat_action_probability=0.0,
        render_mode=render_mode
    )

    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False
    )

    return KerasRLWrapper(env)


def build_model(nb_actions):
    """Build the DQN policy network."""
    model = Sequential()

    model.add(
        Permute(
            (2, 3, 1),
            input_shape=(WINDOW_LENGTH, 84, 84)
        )
    )

    model.add(
        Conv2D(
            32,
            (8, 8),
            strides=(4, 4),
            activation='relu'
        )
    )

    model.add(
        Conv2D(
            64,
            (4, 4),
            strides=(2, 2),
            activation='relu'
        )
    )

    model.add(
        Conv2D(
            64,
            (3, 3),
            strides=(1, 1),
            activation='relu'
        )
    )

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))

    return model


def main():
    """Train the Breakout DQN agent."""
    env = make_env()
    nb_actions = env.action_space.n

    model = build_model(nb_actions)

    memory = SequentialMemory(
        limit=1000000,
        window_length=WINDOW_LENGTH
    )

    epsilon_policy = EpsGreedyQPolicy()

    policy = LinearAnnealedPolicy(
        epsilon_policy,
        attr='eps',
        value_max=1.0,
        value_min=0.1,
        value_test=0.05,
        nb_steps=1000000
    )

    agent = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        policy=policy,
        processor=AtariProcessor(),
        nb_steps_warmup=50000,
        gamma=0.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.0
    )

    agent.compile(
        Adam(learning_rate=0.00025),
        metrics=['mae']
    )

    agent.fit(
        env,
        nb_steps=1000000,
        visualize=False,
        verbose=2
    )

    agent.save_weights(
        'policy.h5',
        overwrite=True
    )

    env.close()


if __name__ == '__main__':
    main()
