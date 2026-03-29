#!/usr/bin/env python3
"""
Module for building and training a decision tree
"""
import numpy as np


class Node:
    """Represent a node in the decision tree"""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initializes the node"""
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
        """Calculates the maximum depth below this node"""
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Counts nodes or leaves below this node"""
        if only_leaves:
            return (self.left_child.count_nodes_below(only_leaves=True) +
                    self.right_child.count_nodes_below(only_leaves=True))
        return (1 + self.left_child.count_nodes_below(only_leaves=False) +
                self.right_child.count_nodes_below(only_leaves=False))

    def get_leaves_below(self):
        """Returns the list of all leaves of the tree below this node"""
        return (self.left_child.get_leaves_below() +
                self.right_child.get_leaves_below())

    def update_bounds_below(self):
        """Recursively computes bounds for each node"""
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
        """Computes and stores the indicator function for the node"""
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
        """Recursive prediction for a single individual"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)

    def left_child_get_prefix(self, text):
        """Adds prefix for left child string representation"""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_get_prefix(self, text):
        """Adds prefix for right child string representation"""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Returns string representation"""
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
    """Represent a leaf in the decision tree"""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        return 1

    def get_leaves_below(self):
        return [self]

    def update_bounds_below(self):
        pass

    def pred(self, x):
        return self.value

    def __str__(self):
        return "-> leaf [value={}]".format(self.value)


class Decision_Tree():
    """Represent a decision tree"""
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
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        return self.root.get_leaves_below()

    def update_bounds(self):
        self.root.update_bounds_below()

    def update_predict(self):
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
        return self.root.pred(x)

    def np_extrema(self, arr):
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
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
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')
        self.fit_node(self.root)
        self.update_predict()
        if verbose == 1:
            print(f"  Training finished.\n"
                  f"    - Depth                     : {self.depth()}\n"
                  f"    - Number of nodes           : {self.count_nodes()}\n"
                  f"    - Number of leaves          : "
                  f"{self.count_nodes(only_leaves=True)}\n"
                  f"    - Accuracy on training data : "
                  f"{self.accuracy(self.explanatory, self.target)}")

    def fit_node(self, node):
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
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size

    def __str__(self):
        return self.root.__str__()
