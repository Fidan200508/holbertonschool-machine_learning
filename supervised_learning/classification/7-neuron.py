#!/usr/bin/env python3
"""Single neuron performing binary classification."""

import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """Defines a single neuron for binary classification."""

    def __init__(self, nx):
        """Initialize the neuron."""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Return the weights."""
        return self.__W

    @property
    def b(self):
        """Return the bias."""
        return self.__b

    @property
    def A(self):
        """Return the activated output."""
        return self.__A

    def forward_prop(self, X):
        """Calculate forward propagation."""
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))

        return self.__A

    def cost(self, Y, A):
        """Calculate logistic regression cost."""
        m = Y.shape[1]

        return -(1 / m) * np.sum(
            Y * np.log(A)
            + (1 - Y) * np.log(1.0000001 - A)
        )

    def evaluate(self, X, Y):
        """Evaluate the neuron's predictions."""
        A = self.forward_prop(X)

        prediction = np.where(
            A >= 0.5,
            1,
            0
        )

        cost = self.cost(Y, A)

        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Perform one pass of gradient descent."""
        m = Y.shape[1]

        dz = A - Y

        dw = (1 / m) * np.matmul(
            dz,
            X.T
        )

        db = (1 / m) * np.sum(dz)

        self.__W = self.__W - alpha * dw
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train the neuron."""
        if type(iterations) is not int:
            raise TypeError(
                "iterations must be an integer"
            )

        if iterations <= 0:
            raise ValueError(
                "iterations must be a positive integer"
            )

        if type(alpha) is not float:
            raise TypeError(
                "alpha must be a float"
            )

        if alpha <= 0:
            raise ValueError(
                "alpha must be positive"
            )

        if verbose or graph:
            if type(step) is not int:
                raise TypeError(
                    "step must be an integer"
                )

            if step <= 0 or step > iterations:
                raise ValueError(
                    "step must be positive and <= iterations"
                )

        costs = []
        steps = []

        for i in range(iterations + 1):
            A = self.forward_prop(X)

            if (verbose or graph) and (
                i % step == 0 or i == iterations
            ):
                cost = self.cost(Y, A)

                if verbose:
                    print(
                        "Cost after {} iterations: {}".format(
                            i,
                            cost
                        )
                    )

                if graph:
                    costs.append(cost)
                    steps.append(i)

            if i < iterations:
                self.gradient_descent(
                    X,
                    Y,
                    A,
                    alpha
                )

        if graph:
            plt.plot(
                steps,
                costs,
                "b"
            )
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
