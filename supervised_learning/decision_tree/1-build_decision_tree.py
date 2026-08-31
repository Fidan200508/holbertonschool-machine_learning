#!/usr/bin/env python3
"""
0-build_decision_tree.py

This module implements a simple Decision Tree in Python.

Classes:
- Node: Represents an internal node in the tree.
- Leaf: Represents a leaf node.
- Decision_Tree: The main tree class with a depth() method.

Each node keeps track of its depth, and max_depth_below() computes
the maximum depth in the subtree from that node.
"""

import numpy as np


class Node:
    """
    Represents an internal node in a decision tree.

    Attributes:
        feature: The index of the feature used for splitting.
        threshold: The threshold value for splitting.
        left_child: The left child Node or Leaf.
        right_child: The right child Node or Leaf.
        is_leaf: Boolean indicating if the node is a leaf.
        is_root: Boolean indicating if the node is the root.
        sub_population: Placeholder for sub-population data.
        depth: Depth of the node in the tree.
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Return the maximum depth of this node and all its descendants.
        """
        left_depth = (self.left_child.max_depth_below()
                      if self.left_child else self.depth)
        right_depth = (self.right_child.max_depth_below()
                       if self.right_child else self.depth)
        return max(self.depth, left_depth, right_depth)



    def count_nodes_below(self, only_leaves=False):
        """Count nodes below the current node."""
        left = self.left_child.count_nodes_below(
            only_leaves=only_leaves
        )
        right = self.right_child.count_nodes_below(
            only_leaves=only_leaves
        )

        if only_leaves:
            return left + right

        return 1 + left + right
class Leaf(Node):
    """
    Represents a leaf node in a decision tree.

    Attributes:
        value: The value stored in the leaf.
        depth: Depth of the leaf in the tree.
    """

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Return the depth of the leaf.
        """
        return self.depth



    def count_nodes_below(self, only_leaves=False):
        """Count this leaf."""
        return 1
class Decision_Tree:
    """
    Represents a decision tree.

    Attributes:
        root: The root Node of the tree.
        max_depth: Maximum allowed depth of the tree.
        min_pop: Minimum population for splitting.
        split_criterion: Criterion used for splitting.
        rng: Random number generator.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Return the maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count nodes in the decision tree."""
        return self.root.count_nodes_below(
            only_leaves=only_leaves
        )
