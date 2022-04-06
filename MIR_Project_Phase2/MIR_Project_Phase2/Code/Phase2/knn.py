import numpy as np
from sklearn.metrics import pairwise_distances
from collections import Counter


class KNN:
    def __init__(self, n_neighbours):
        self.k = n_neighbours
        self.train_X = None
        self.train_y = None

    def fit(self, X, y):
        self.train_X = X
        self.train_y = np.asarray(y)

    def predict(self, X, tags=None):
        predictions = np.empty((len(X),),
                               dtype=int)
        dist = pairwise_distances(X, self.train_X)
        for i, row in enumerate(dist):
            knn_indices = np.argsort(row)[0:self.k]
            pr_tag = Counter(self.train_y[knn_indices]).most_common(1)[0][0]
            predictions[i] = pr_tag
        return predictions
