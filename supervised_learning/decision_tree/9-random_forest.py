#!/usr/bin/env python3
"""Random forest module."""

import numpy as np

Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest:
    """Represents a random forest."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initialize the random forest."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """Predict classes using majority vote."""
        predictions = np.array([
            predict(explanatory)
            for predict in self.numpy_preds
        ])

        def mode(column):
            """Return the most frequent value."""
            values, counts = np.unique(
                column,
                return_counts=True
            )
            return values[np.argmax(counts)]

        return np.apply_along_axis(
            mode,
            axis=0,
            arr=predictions
        )

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Train the random forest."""
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []

        depths = []
        nodes = []
        leaves = []
        accuracies = []

        for i in range(n_trees):
            tree = Decision_Tree(
                max_depth=self.max_depth,
                min_pop=self.min_pop,
                seed=self.seed + i
            )

            tree.fit(explanatory, target)

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

            accuracies.append(
                tree.accuracy(
                    tree.explanatory,
                    tree.target
                )
            )

        if verbose == 1:
            print(
                "  Training finished.\n"
                "    - Mean depth                     : "
                "{}\n"
                "    - Mean number of nodes           : "
                "{}\n"
                "    - Mean number of leaves          : "
                "{}\n"
                "    - Mean accuracy on training data : "
                "{}\n"
                "    - Accuracy of the forest on td   : "
                "{}".format(
                    np.array(depths).mean(),
                    np.array(nodes).mean(),
                    np.array(leaves).mean(),
                    np.array(accuracies).mean(),
                    self.accuracy(
                        self.explanatory,
                        self.target
                    )
                )
            )

    def accuracy(self, test_explanatory, test_target):
        """Calculate the accuracy of the forest."""
        predictions = self.predict(
            test_explanatory
        )

        return (
            np.sum(
                np.equal(
                    predictions,
                    test_target
                )
            )
            / test_target.size
        )
