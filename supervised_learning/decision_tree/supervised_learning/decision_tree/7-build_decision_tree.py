#!/usr/bin/env python3
"""
Module to build and train a decision tree
"""
import numpy as np


class Node:
    """Class representing a node in a decision tree"""
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
        if self.is_leaf:
            return self.depth
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Counts the nodes below this node"""
        if only_leaves:
            return (self.left_child.count_nodes_below(only_leaves=True) +
                    self.right_child.count_nodes_below(only_leaves=True))
        return (1 + self.left_child.count_nodes_below(only_leaves=False) +
                self.right_child.count_nodes_below(only_leaves=False))

    def get_leaves_below(self):
        """Returns the list of all leaves of the tree"""
        if self.is_leaf:
            return [self]
        return (self.left_child.get_leaves_below() +
                self.right_child.get_leaves_below())

    def update_bounds_below(self):
        """Recursively compute lower and upper bounds for each node"""
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()

        self.left_child.lower[self.feature] = self.threshold
        self.right_child.upper[self.feature] = self.threshold

        self.left_child.update_bounds_below()
        self.right_child.update_bounds_below()

    def update_indicator(self):
        """Computes the indicator function from bounds"""
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
        """Predicts the value for a single individual"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)

    def left_child_get_prefix(self, text):
        """Prefix helper for string representation"""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_get_prefix(self, text):
        """Prefix helper for string representation"""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """String representation of the node"""
        label = "root" if self.is_root else "node"
        out = f"{label} [feature={self.feature}, threshold={self.threshold}]\n"
        out += self.left_child_get_prefix(self.left_child.__str__())
        out += self.right_child_get_prefix(self.right_child.__str__())
        return out


class Leaf(Node):
    """Class representing a leaf in a decision tree"""
    def __init__(self, value, depth=None):
        """Initializes the leaf"""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the depth of the leaf"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Returns 1 for a leaf"""
        return 1

    def get_leaves_below(self):
        """Returns the leaf itself"""
        return [self]

    def update_bounds_below(self):
        """Leaves are terminal"""
        pass

    def pred(self, x):
        """Returns the value of the leaf"""
        return self.value

    def __str__(self):
        """String representation of the leaf"""
        return f"leaf [value={self.value}]"


class Decision_Tree:
    """Class representing a decision tree"""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initializes the decision tree"""
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
        """Returns the maximum depth of the tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Counts the nodes in the tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """Returns the list of all leaves of the tree"""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Triggers the bounds update for the tree"""
        self.root.update_bounds_below()

    def update_predict(self):
        """Updates the prediction lambda"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(np.array([leaf.value * leaf.indicator(A)
                                                  for leaf in leaves]), axis=0)

    def pred(self, x):
        """Root prediction helper"""
        return self.root.pred(x)

    def np_extrema(self, arr):
        """Returns min and max of array"""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Finds a random split for the node"""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            f_min, f_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = f_max - f_min
        x = self.rng.uniform()
        threshold = (1 - x) * f_min + x * f_max
        return feature, threshold

    def fit(self, explanatory, target, verbose=0):
        """Fits the tree to the data"""
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
            print(f"""  Training finished.
- Depth                     : {self.depth()}
- Number of nodes           : {self.count_nodes()}
- Number of leaves          : {self.count_nodes(only_leaves=True)}
- Accuracy on training data : {self.accuracy(self.explanatory, self.target)}""")

    def fit_node(self, node):
        """Recursively fit the nodes"""
        node.feature, node.threshold = self.split_criterion(node)
        left_mask = self.explanatory[:, node.feature] > node.threshold
        right_mask = self.explanatory[:, node.feature] <= node.threshold
        left_population = np.logical_and(node.sub_population, left_mask)
        right_population = np.logical_and(node.sub_population, right_mask)

        def is_leaf_condition(pop, depth):
            if np.sum(pop) < self.min_pop or depth >= self.max_depth:
                return True
            if np.unique(self.target[pop]).size <= 1:
                return True
            return False

        if is_leaf_condition(left_population, node.depth + 1):
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        if is_leaf_condition(right_population, node.depth + 1):
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Helper to create a leaf child"""
        targets = self.target[sub_population]
        classes, counts = np.unique(targets, return_counts=True)
        value = classes[np.argmax(counts)]
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Helper to create an internal node child"""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Calculates prediction accuracy"""
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size
