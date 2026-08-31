#!/usr/bin/env python3
"""Isolation random tree module."""

import numpy as np

Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree:
    """Represents an isolation random tree."""

    def __init__(self, max_depth=10, seed=0, root=None):
        """Initialize the isolation tree."""
        self.rng = np.random.default_rng(seed)

        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)

        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """Return string representation of the tree."""
        return self.root.__str__()

    def depth(self):
        """Return maximum depth of the tree."""
        leaves = self.get_leaves()

        return max(
            leaf.depth
            for leaf in leaves
        )

    def count_nodes(self, only_leaves=False):
        """Count nodes in the tree."""
        return self.root.count_nodes_below(
            only_leaves=only_leaves
        )

    def update_bounds(self):
        """Update bounds of the tree."""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Return all leaves of the tree."""
        return self.root.get_leaves_below()

    def update_predict(self):
        """Update prediction function."""
        self.update_bounds()

        leaves = self.get_leaves()

        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: np.sum(
            np.array([
                leaf.value * leaf.indicator(A)
                for leaf in leaves
            ]),
            axis=0
        )

    def np_extrema(self, arr):
        """Return minimum and maximum values."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Select a random feature and threshold."""
        diff = 0

        while diff == 0:
            feature = self.rng.integers(
                0,
                self.explanatory.shape[1]
            )

            values = self.explanatory[:, feature][
                node.sub_population
            ]

            feature_min, feature_max = self.np_extrema(
                values
            )

            diff = feature_max - feature_min

        x = self.rng.uniform()

        threshold = (
            (1 - x) * feature_min
            + x * feature_max
        )

        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Create a leaf whose value is its depth."""
        leaf_child = Leaf(node.depth + 1)

        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        leaf_child.subpopulation = sub_population

        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create an internal node child."""
        child = Node()

        child.depth = node.depth + 1
        child.sub_population = sub_population

        return child

    def fit_node(self, node):
        """Recursively fit the isolation tree."""
        node.feature, node.threshold = (
            self.random_split_criterion(node)
        )

        feature_values = self.explanatory[
            :, node.feature
        ]

        left_population = np.logical_and(
            node.sub_population,
            feature_values > node.threshold
        )

        right_population = np.logical_and(
            node.sub_population,
            feature_values <= node.threshold
        )

        is_left_leaf = (
            np.sum(left_population) <= self.min_pop
            or node.depth + 1 == self.max_depth
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(
                node,
                left_population
            )
        else:
            node.left_child = self.get_node_child(
                node,
                left_population
            )

            self.fit_node(
                node.left_child
            )

        is_right_leaf = (
            np.sum(right_population) <= self.min_pop
            or node.depth + 1 == self.max_depth
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(
                node,
                right_population
            )
        else:
            node.right_child = self.get_node_child(
                node,
                right_population
            )

            self.fit_node(
                node.right_child
            )

    def fit(self, explanatory, verbose=0):
        """Train the isolation random tree."""
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory

        self.root.sub_population = np.ones(
            explanatory.shape[0],
            dtype='bool'
        )

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(
                "  Training finished.\n"
                "    - Depth                     : {}\n"
                "    - Number of nodes           : {}\n"
                "    - Number of leaves          : {}".format(
                    self.depth(),
                    self.count_nodes(),
                    self.count_nodes(
                        only_leaves=True
                    )
                )
            )
