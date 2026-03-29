#!/usr/bin/env python3
"""
Module for building and training a decision tree model.
Contains Node, Leaf, and Decision_Tree classes.
"""
import numpy as np


class Node:
    """
    Represent an internal node in the decision tree.
    Stores splitting information and references to children.
    """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initializes the node with features, threshold, and children.
        """
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
        """
        Recursively calculates the maximum depth reached by any node
        below this specific internal node.
        """
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the number of nodes or leaves in the subtree
        starting from this node.
        """
        if only_leaves:
            return (self.left_child.count_nodes_below(only_leaves=True) +
                    self.right_child.count_nodes_below(only_leaves=True))
        return (1 + self.left_child.count_nodes_below(only_leaves=False) +
                self.right_child.count_nodes_below(only_leaves=False))

    def get_leaves_below(self):
        """
        Traverses the tree to collect and return a list of all
        leaf objects located below this node.
        """
        return (self.left_child.get_leaves_below() +
                self.right_child.get_leaves_below())

    def update_bounds_below(self):
        """
        Recursively updates the lower and upper feature bounds
        for all children based on the current node's threshold.
        """
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
        """
        Defines and stores an indicator function that determines
        if individuals fall within this node's boundaries.
        """
        def is_large_enough(x):
            """Checks if features are greater than lower bounds."""
            return np.all(np.array([np.greater(x[:, key], self.lower[key])
                                    for key in self.lower.keys()]), axis=0)

        def is_small_enough(x):
            """Checks if features are less than or equal to upper bounds."""
            return np.all(np.array([np.less_equal(x[:, key], self.upper[key])
                                    for key in self.upper.keys()]), axis=0)

        self.indicator = lambda x: np.all(np.array([is_large_enough(x),
                                                    is_small_enough(x)]),
                                          axis=0)

    def pred(self, x):
        """
        Predicts the class for a single data point by traversing
        the tree recursively.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)

    def left_child_get_prefix(self, text):
        """
        Formatting helper to add visual prefixes to the string
        representation of a left child branch.
        """
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_get_prefix(self, text):
        """
        Formatting helper to add visual prefixes to the string
        representation of a right child branch.
        """
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """
        Returns a formatted string representing the node's
        feature and threshold for tree visualization.
        """
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
    """
    Represent a terminal leaf in the decision tree.
    Holds the final predicted value for its sub-population.
    """
    def __init__(self, value, depth=None):
        """
        Initializes the leaf with a prediction value and its depth.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of this leaf. Since it has no children,
        the maximum depth below it is just its own depth.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Returns 1, as the leaf is a single unit in the node count,
        regardless of whether we are counting all nodes or just leaves.
        """
        return 1

    def get_leaves_below(self):
        """
        Returns a list containing only this leaf instance,
        serving as the base case for leaf collection recursion.
        """
        return [self]

    def update_bounds_below(self):
        """
        Does nothing. Leaves are terminal points in the tree
        and do not have children to update boundaries for.
        """
        pass

    def pred(self, x):
        """
        Returns the prediction value stored in this leaf
        for the given input individual.
        """
        return self.value

    def __str__(self):
        """
        Returns a formatted string representing the leaf and
        its stored prediction value.
        """
        return "-> leaf [value={}]".format(self.value)


class Decision_Tree():
    """
    Represent the entire decision tree model.
    Handles training, prediction, and tree property calculations.
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initializes the decision tree with hyperparameters and root.
        """
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
        """Calculates the total depth of the decision tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Calculates total nodes or leaf count for the entire tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """Retrieves a list of every leaf currently in the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Triggers the boundary update process from the root."""
        self.root.update_bounds_below()

    def update_predict(self):
        """Generates the vectorized prediction function for the tree."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_func(A):
            """Vectorized prediction internal logic."""
            results = np.zeros(A.shape[0])
            for leaf in leaves:
                results[leaf.indicator(A)] = leaf.value
            return results
        self.predict = predict_func

    def pred(self, x):
        """Predicts class for one individual via root recursion."""
        return self.root.pred(x)

    def np_extrema(self, arr):
        """Finds the minimum and maximum values of a numpy array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Randomly selects a feature and threshold for a node split."""
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

    def fit(self, explanatory, target, verbose=0):
        """Trains the decision tree on a provided dataset."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
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
        """Recursively partitions data to grow the decision tree."""
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
        """Creates a leaf child and assigns the majority class value."""
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
        """Creates an internal node child for the tree."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Computes the accuracy of the tree on a test dataset."""
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size

    def __str__(self):
        """Returns the string representation of the root node."""
        return self.root.__str__()
