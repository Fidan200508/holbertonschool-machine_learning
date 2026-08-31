#!/usr/bin/env python3
"""Normal distribution module."""


class Normal:
    """Represents a normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize the normal distribution."""
        if data is None:
            if stddev <= 0:
                raise ValueError(
                    "stddev must be a positive value"
                )

            self.mean = float(mean)
            self.stddev = float(stddev)

        else:
            if not isinstance(data, list):
                raise TypeError(
                    "data must be a list"
                )

            if len(data) < 2:
                raise ValueError(
                    "data must contain multiple values"
                )

            self.mean = float(
                sum(data) / len(data)
            )

            variance = sum(
                (x - self.mean) ** 2
                for x in data
            ) / len(data)

            self.stddev = float(
                variance ** 0.5
            )

    def z_score(self, x):
        """Calculate the z-score of x."""
        return (
            (x - self.mean)
            / self.stddev
        )

    def x_value(self, z):
        """Calculate the x-value of a z-score."""
        return (
            z * self.stddev
            + self.mean
        )

    def pdf(self, x):
        """Calculate the PDF for a given x-value."""
        pi = 3.1415926536
        e = 2.7182818285

        coefficient = (
            1
            / (
                self.stddev
                * (2 * pi) ** 0.5
            )
        )

        exponent = (
            -0.5
            * (
                (x - self.mean)
                / self.stddev
            ) ** 2
        )

        return coefficient * (e ** exponent)
