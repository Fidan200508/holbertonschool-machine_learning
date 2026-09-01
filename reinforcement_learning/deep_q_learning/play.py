#!/usr/bin/env python3
"""Play Atari Breakout using a trained DQN agent."""

from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy

from train import AtariProcessor
from train import WINDOW_LENGTH
from train import build_model
from train import make_env


def main():
    """Load the trained policy and play Breakout."""
    env = make_env(render_mode='human')
    nb_actions = env.action_space.n

    model = build_model(nb_actions)

    memory = SequentialMemory(
        limit=1000000,
        window_length=WINDOW_LENGTH
    )

    greedy_policy = GreedyQPolicy()

    agent = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        policy=greedy_policy,
        test_policy=greedy_policy,
        processor=AtariProcessor()
    )

    agent.compile(
        Adam(learning_rate=0.00025),
        metrics=['mae']
    )

    agent.load_weights('policy.h5')

    agent.test(
        env,
        nb_episodes=5,
        visualize=True
    )

    env.close()


if __name__ == '__main__':
    main()
