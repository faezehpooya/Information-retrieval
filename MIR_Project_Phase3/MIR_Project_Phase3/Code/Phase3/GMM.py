import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import seaborn as sns


class GMM:
    def __init__(self, k=3, vectorizer_name='', max_iters=250, covariance_type='full'):  # covariance_type='diag'
        self.method_name = 'GMM'
        self.vectorizer_name = vectorizer_name
        self.k = k  # number of Guassians/clusters
        self.max_iters = max_iters
        self.model = GaussianMixture(n_components=self.k,
                                     max_iter=self.max_iters,
                                     covariance_type=covariance_type)

    def fit(self, X):
        self.x = X

        self.model.fit(X)
        self.centers = self.model.means_
        self.predictions = self.predict(X)
        return

    def predict(self, X):
        return self.model.predict(X)

    def get_cooccurrence_matrix(self, _y, show=False):
        cooccurrence_matrix = np.array([[0 for i in range(self.k)] for j in range(self.k)])
        predictions = self.predictions
        for i in range(len(_y)):
            cooccurrence_matrix[_y[i]][predictions[i]] += 1

        if show:
            df = pd.DataFrame(cooccurrence_matrix, columns=['Cluster ' + str(i) for i in range(self.k)],
                              index=['Class ' + str(i) for i in range(self.k)])
            print(df)

        return cooccurrence_matrix

    def plot(gmm, label=True, ax=None):
        #     plt.figure(figsize = (10,8))
        ax = ax or plt.gca()
        labels = gmm.predictions

        pca = PCA(n_components=2)
        dim_reduced_X = pca.fit_transform(gmm.x)
        X = dim_reduced_X

        # # compute centers as point of highest density of distribution
        # centers = gmm.centers

        if label:
            colors = (sns.color_palette("bright") + sns.color_palette("dark") +
                      sns.color_palette("Set3") + sns.color_palette("pastel") +
                      sns.color_palette("muted"))
            cluster_colors = [colors[i] for i in labels]
            ax.scatter(X[:, 0], X[:, 1], c=cluster_colors, s=5, cmap='viridis', zorder=2)
            centers = pca.fit_transform(gmm.centers)
            plt.scatter(centers[:, 0], centers[:, 1], c='black', s=50, zorder=5, marker='x')

        else:
            ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
        #     ax.axis('equal')

        plt.title(gmm.method_name + ' - ' + gmm.vectorizer_name + ' (K = {})'.format(gmm.k))
        plt.show()
