#!/usr/bin/env python3
"""Isolation random forest module."""

import numpy as np

Isolation_Random_Tree = (
    __import__('10-isolation_tree').Isolation_Random_Tree
)


class Isolation_Random_Forest:
    """Represents an isolation random forest."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initialize the isolation forest."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """Return mean isolation depth across all trees."""
        predictions = np.array([
            f(explanatory)
            for f in self.numpy_preds
        ])

        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """Train the isolation forest."""
        self.explanatory = explanatory
        self.numpy_preds = []

        depths = []
        nodes = []
        leaves = []

        for i in range(n_trees):
            tree = Isolation_Random_Tree(
                max_depth=self.max_depth,
                seed=self.seed + i
            )

            tree.fit(explanatory)

            self.numpy_preds.append(
                tree.predict
            )

            depths.append(
                tree.depth()
            )

            nodes.append(
                tree.count_nodes()
            )

            leaves.append(
                tree.count_nodes(
                    only_leaves=True
                )
            )

        if verbose == 1:
            print(
                "  Training finished.\n"
                "    - Mean depth                     : {}\n"
                "    - Mean number of nodes           : {}\n"
                "    - Mean number of leaves          : {}".format(
                    np.array(depths).mean(),
                    np.array(nodes).mean(),
                    np.array(leaves).mean()
                )
            )

    def suspects(self, explanatory, n_suspects):
        """Return rows with the smallest mean isolation depth."""
        depths = self.predict(explanatory)

        indices = np.argsort(depths)[:n_suspects]

        return explanatory[indices], depths[indices]
