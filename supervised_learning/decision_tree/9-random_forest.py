#!/usr/bin/env python3
"""
Module defining the Random_Forest class.
This class implements a forest of decision trees that uses majority voting.
"""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """
    Class representing a Random Forest model composed of multiple
    decision trees to improve stability and prediction accuracy.
    """
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
        Initializes the Random Forest with hyperparameters.
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """
        Predicts the class for each individual in the provided dataset
        by collecting predictions from all trees in the forest and
        returning the majority vote (mode) for each individual.
        """
        # Collect predictions from every tree: shape (n_trees, n_individuals)
        all_preds = np.array([f(explanatory) for f in self.numpy_preds])

        # Calculate the mode (most frequent value) for each individual (column)
        # np.apply_along_axis is used to keep the operation vectorized
        return np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(), 0, all_preds
        )

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """
        Trains the Random Forest by fitting multiple decision trees
        on the training dataset and storing their prediction functions.
        """
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []

        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth,
                              min_pop=self.min_pop,
                              seed=self.seed + i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))

        if verbose == 1:
            print("  Training finished.")
            print("    - Mean depth                     : {}".format(
                np.array(depths).mean()))
            print("    - Mean number of nodes           : {}".format(
                np.array(nodes).mean()))
            print("    - Mean number of leaves          : {}".format(
                np.array(leaves).mean()))
            print("    - Mean accuracy on training data : {}".format(
                np.array(accuracies).mean()))
            print("    - Accuracy of the forest on td   : {}".format(
                self.accuracy(self.explanatory, self.target)))

    def accuracy(self, test_explanatory, test_target):
        """
        Calculates the accuracy of the forest's predictions
        against the ground truth target labels.
        """
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size
