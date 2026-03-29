#!/usr/bin/env python3
"""
Module for building and training a decision tree using Gini Impurity.
This module contains the logic for finding the optimal mathematical split.
"""
import numpy as np


class Node:
    """Represent an internal node in the decision tree."""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initializes the node with split criteria and children."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth
        self.lower = None
        self.upper = None
        self.indicator = None

    def max_depth_below(self):
        """Calculates the maximum depth reached in this branch."""
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Counts the nodes or leaves in the subtree."""
        if only_leaves:
            return (self.left_child.count_nodes_below(only_leaves=True) +
                    self.right_child.count_nodes_below(only_leaves=True))
        return (1 + self.left_child.count_nodes_below(only_leaves=False) +
                self.right_child.count_nodes_below(only_leaves=False))

    def get_leaves_below(self):
        """Returns the list of all leaves of the tree below this node."""
        return (self.left_child.get_leaves_below() +
                self.right_child.get_leaves_below())

    def update_bounds_below(self):
        """Recursively computes the upper and lower bounds for each node."""
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}
        for child in [self.left_child, self.right_child]:
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()
        self.left_child.lower[self.feature] = self.threshold
        self.right_child.upper[self.feature] = self.threshold
        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Computes and stores the indicator function for the node."""
        def is_large_enough(x):
            return np.all(np.array([np.greater(x[:, key], self.lower[key])
                                    for key in self.lower.keys()]), axis=0)

        def is_small_enough(x):
            return np.all(np.array([np.less_equal(x[:, key], self.upper[key])
                                    for key in self.upper.keys()]), axis=0)

        self.indicator = lambda x: np.all(np.array([is_large_enough(x),
                                                    is_small_enough(x)]),
                                          axis=0)

    def pred(self, x):
        """Recursive prediction for a single individual."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)

    def left_child_get_prefix(self, text):
        """Adds prefix for left child string representation."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_get_prefix(self, text):
        """Adds prefix for right child string representation."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Returns the string representation of the node."""
        if self.is_root:
            out = "root [feature={}, threshold={}]\n".format(self.feature,
                                                             self.threshold)
        else:
            out = "node [feature={}, threshold={}]\n".format(self.feature,
                                                             self.threshold)
        out += self.left_child_get_prefix(self.left_child.__str__())
        out += self.right_child_get_prefix(self.right_child.__str__())
        return out


class Leaf(Node):
    """Represent a leaf in the decision tree."""
    def __init__(self, value, depth=None):
        """Initializes the leaf with its class value and depth."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the depth of the leaf node."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Returns 1 as the leaf is a single unit."""
        return 1

    def get_leaves_below(self):
        """Returns the leaf itself in a list."""
        return [self]

    def update_bounds_below(self):
        """Leaves do not have children, so no bounds to update."""
        pass

    def pred(self, x):
        """Returns the prediction value of the leaf."""
        return self.value

    def __str__(self):
        """Returns the string representation of the leaf."""
        return "-> leaf [value={}]".format(self.value)


class Decision_Tree():
    """Represent a decision tree model with training capabilities."""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initializes the decision tree."""
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
        """Returns the maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Counts the nodes in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """Returns the list of all leaves of the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Updates the bounds for all nodes in the tree."""
        self.root.update_bounds_below()

    def update_predict(self):
        """Computes the prediction function."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_func(A):
            results = np.zeros(A.shape[0])
            for leaf in leaves:
                results[leaf.indicator(A)] = leaf.value
            return results
        self.predict = predict_func

    def pred(self, x):
        """Recursive prediction from the root."""
        return self.root.pred(x)

    def np_extrema(self, arr):
        """Helper to find min and max of array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Random split logic."""
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

    def possible_thresholds(self, node, feature):
        """Calculates midpoints between unique sorted values."""
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """Computes best Gini split for a single feature."""
        x = self.explanatory[node.sub_population, feature]
        y = self.target[node.sub_population]
        t = self.possible_thresholds(node, feature)
        n = y.size
        classes = np.unique(y)

        # 3D tensor: [individuals, thresholds, classes]
        greater_mask = x[:, np.newaxis] > t
        class_mask = y[:, np.newaxis] == classes
        Left_F = greater_mask[:, :, np.newaxis] & class_mask[:, np.newaxis, :]

        # Left Child Stats
        n_left = np.sum(greater_mask, axis=0)
        c_left = np.sum(Left_F, axis=0)
        # Avoid division by zero
        p_left = np.divide(c_left, n_left[:, np.newaxis],
                           out=np.zeros_like(c_left, dtype=float),
                           where=n_left[:, np.newaxis] != 0)
        gini_left = 1 - np.sum(p_left**2, axis=1)

        # Right Child Stats
        n_right = n - n_left
        c_right = np.sum(class_mask, axis=0) - c_left
        p_right = np.divide(c_right, n_right[:, np.newaxis],
                            out=np.zeros_like(c_right, dtype=float),
                            where=n_right[:, np.newaxis] != 0)
        gini_right = 1 - np.sum(p_right**2, axis=1)

        # Weighted Gini Average
        gini_avg = (n_left * gini_left + n_right * gini_right) / n
        best_idx = np.argmin(gini_avg)

        return t[best_idx], gini_avg[best_idx]

    def Gini_split_criterion(self, node):
        """Finds the best split across all features."""
        X = np.array([self.Gini_split_criterion_one_feature(node, i)
                      for i in range(self.explanatory.shape[1])])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]

    def fit(self, explanatory, target, verbose=0):
        """Trains the decision tree on the given data."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')
        self.fit_node(self.root)
        self.update_predict()
        if verbose == 1:
            print("  Training finished.")
            print("- Depth                     : {}".format(self.depth()))
            print("- Number of nodes           : {}".format(self.count_nodes()))
            print("- Number of leaves          : {}".format(
                self.count_nodes(only_leaves=True)))
            print("- Accuracy on training data : {}".format(
                self.accuracy(self.explanatory, self.target)))

    def fit_node(self, node):
        """Recursive node training."""
        node.feature, node.threshold = self.split_criterion(node)
        left_pop = np.logical_and(
            node.sub_population,
            self.explanatory[:, node.feature] > node.threshold
        )
        right_pop = np.logical_and(
            node.sub_population,
            self.explanatory[:, node.feature] <= node.threshold
        )
        is_left_leaf = (
            node.depth + 1 == self.max_depth or
            np.sum(left_pop) <= self.min_pop or
            np.unique(self.target[left_pop]).size == 1
        )
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_pop)
        else:
            node.left_child = self.get_node_child(node, left_pop)
            self.fit_node(node.left_child)
        is_right_leaf = (
            node.depth + 1 == self.max_depth or
            np.sum(right_pop) <= self.min_pop or
            np.unique(self.target[right_pop]).size == 1
        )
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_pop)
        else:
            node.right_child = self.get_node_child(node, right_pop)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Creates a leaf child node."""
        classes = self.target[sub_population]
        if classes.size == 0:
            value = 0
        else:
            value = np.bincount(classes).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Creates an internal node child."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Calculates accuracy on test data."""
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size

    def __str__(self):
        """String representation of root."""
        return self.root.__str__()
