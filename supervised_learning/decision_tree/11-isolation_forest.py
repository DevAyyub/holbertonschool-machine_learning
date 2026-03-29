#!/usr/bin/env python3
"""
Module for building an Isolation Random Forest.
Used for unsupervised anomaly detection by averaging isolation depths.
"""
import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """
    Class representing an Isolation Random Forest model.
    Detects outliers by finding individuals that are isolated quickly.
    """
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
        Initializes the Isolation Random Forest with hyperparameters.
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """
        Calculates the mean isolation depth for each individual across
        all trees in the forest.
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """
        Trains the forest by fitting multiple isolation random trees.
        """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(max_depth=self.max_depth,
                                      seed=self.seed + i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            print("  Training finished.")
            print("    - Mean depth                     : {}".format(
                np.array(depths).mean()))
            print("    - Mean number of nodes           : {}".format(
                np.array(nodes).mean()))
            print("    - Mean number of leaves          : {}".format(
                np.array(leaves).mean()))

    def suspects(self, explanatory, n_suspects):
        """
        Identifies and returns the individuals most likely to be outliers
        based on their mean isolation depth across the forest.
        """
        depths = self.predict(explanatory)
        # Get indices of n smallest depths
        indices = np.argsort(depths)
        suspect_indices = indices[:n_suspects]

        return explanatory[suspect_indices], depths[suspect_indices]
