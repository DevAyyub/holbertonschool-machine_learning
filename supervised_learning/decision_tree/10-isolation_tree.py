#!/usr/bin/env python3
"""
Module for building an Isolation Random Tree.
Used for anomaly detection by isolating outliers in shallow leaves.
"""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """
    Class representing an Isolation Random Tree for outlier detection.
    """
    def __init__(self, max_depth=10, seed=0, root=None):
        """Initializes the isolation tree with depth and seed."""
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
        """Returns the string representation of the root node."""
        return self.root.__str__()

    def depth(self):
        """Returns the maximum depth reached in the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Counts total nodes or leaves in the isolation tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """Triggers boundary computation from the root node."""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Collects all leaf nodes currently in the tree."""
        return self.root.get_leaves_below()

    def update_predict(self):
        """Updates the vectorized prediction function for leaf depths."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_func(A):
            """Returns the depth of the leaf for each individual."""
            results = np.zeros(A.shape[0])
            for leaf in leaves:
                results[leaf.indicator(A)] = leaf.value
            return results
        self.predict = predict_func

    def np_extrema(self, arr):
        """Finds the min and max values of a numpy array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Selects a random feature and threshold for isolation."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            f_min, f_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
            )
            diff = f_max - f_min
        x = self.rng.uniform()
        threshold = (1 - x) * f_min + x * f_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Creates a leaf child where value is the depth of the leaf."""
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Creates an internal node child for further isolation."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Recursively partitions data until population is 1 or max depth."""
        node.feature, node.threshold = self.random_split_criterion(node)

        left_pop = np.logical_and(
            node.sub_population,
            self.explanatory[:, node.feature] > node.threshold
        )
        right_pop = np.logical_and(
            node.sub_population,
            self.explanatory[:, node.feature] <= node.threshold
        )

        # Is left node a leaf ?
        is_left_leaf = (
            node.depth + 1 == self.max_depth or
            np.sum(left_pop) <= self.min_pop
        )
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_pop)
        else:
            node.left_child = self.get_node_child(node, left_pop)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = (
            node.depth + 1 == self.max_depth or
            np.sum(right_pop) <= self.min_pop
        )
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_pop)
        else:
            node.right_child = self.get_node_child(node, right_pop)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Trains the isolation tree by splitting data randomly."""
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print("  Training finished.")
            print("    - Depth                     : {}".format(self.depth()))
            print("    - Number of nodes           : {}".format(
                self.count_nodes()))
            print("    - Number of leaves          : {}".format(
                self.count_nodes(only_leaves=True)))
