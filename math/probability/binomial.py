#!/usr/bin/env python3
"""Binomial distribution module."""


class Binomial:
    """Represents a binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize the binomial distribution."""
        if data is None:
            if n <= 0:
                raise ValueError(
                    "n must be a positive value"
                )

            if p <= 0 or p >= 1:
                raise ValueError(
                    "p must be greater than 0 and less than 1"
                )

            self.n = int(n)
            self.p = float(p)

        else:
            if not isinstance(data, list):
                raise TypeError(
                    "data must be a list"
                )

            if len(data) < 2:
                raise ValueError(
                    "data must contain multiple values"
                )

            mean = sum(data) / len(data)

            variance = sum(
                (x - mean) ** 2
                for x in data
            ) / len(data)

            p = 1 - (variance / mean)
            n = round(mean / p)
            p = mean / n

            self.n = int(n)
            self.p = float(p)

    def pmf(self, k):
        """Calculate the PMF for a given number of successes."""
        k = int(k)

        if k < 0 or k > self.n:
            return 0

        def factorial(x):
            """Calculate factorial of x."""
            result = 1

            for i in range(1, x + 1):
                result *= i

            return result

        combination = (
            factorial(self.n)
            / (
                factorial(k)
                * factorial(self.n - k)
            )
        )

        return (
            combination
            * (self.p ** k)
            * ((1 - self.p) ** (self.n - k))
        )

    def cdf(self, k):
        """Calculate the CDF for a given number of successes."""
        k = int(k)

        if k < 0:
            return 0

        if k > self.n:
            return 0

        return sum(
            self.pmf(i)
            for i in range(k + 1)
        )
