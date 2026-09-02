#!/usr/bin/env python3
"""Training function for Monte Carlo policy gradient."""

import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """Train an agent using the Monte Carlo policy gradient algorithm."""
    weight = np.random.rand(
        env.observation_space.shape[0],
        env.action_space.n
    )
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        gradients = []
        rewards = []
        done = False

        while not done:
            action, gradient = policy_gradient(state, weight)

            next_state, reward, terminated, truncated, _ = env.step(action)

            gradients.append(gradient)
            rewards.append(reward)

            state = next_state
            done = terminated or truncated

        score = sum(rewards)
        scores.append(score)

        for i, gradient in enumerate(gradients):
            discounted_reward = 0

            for j, reward in enumerate(rewards[i:]):
                discounted_reward += reward * (gamma ** j)

            weight += alpha * gradient * discounted_reward

        print("Episode: {} Score: {}".format(episode, score))

    return scores
