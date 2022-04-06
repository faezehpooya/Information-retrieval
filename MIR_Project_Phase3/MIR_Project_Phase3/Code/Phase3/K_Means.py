import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns


class Kmeans:

    def __init__(self, k=3, vectorizer_name='', max_iters=1000, init='k-means++'):
        self.method_name = 'KMeans'
        self.vectorizer_name = vectorizer_name
        self.k = k
        self.max_iters = max_iters
        self.model = KMeans(n_clusters=k, init=init, max_iter=max_iters)

    def fit(self, x):
        self.x = x
        self.data_size = x.shape[0]
        self.feature_size = x.shape[1]

        self.model.fit(x)
        self.centers = self.model.cluster_centers_
        self.predictions = self.model.predict(x)

    def predict(self, x):
        return self.model.predict(x)

    def get_cooccurrence_matrix(self, _y, show=False):
        cooccurrence_matrix = np.array([[0 for i in range(self.k)] for j in range(self.k)])
        for i in range(len(_y)):
            cooccurrence_matrix[_y[i]][self.predictions[i]] += 1

        if show:
            df = pd.DataFrame(cooccurrence_matrix, columns=['Cluster ' + str(i) for i in range(self.k)],
                              index=['Class ' + str(i) for i in range(self.k)])
            print(df)

        return cooccurrence_matrix

    def plot(self, ax=None):
        labels = self.predictions
        # colors = ['b', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        colors = (sns.color_palette("bright") + sns.color_palette("dark") +
                  sns.color_palette("Set3") + sns.color_palette("pastel") +
                  sns.color_palette("muted"))
        cluster_colors = [colors[i] for i in labels]

        pca = PCA(n_components=2)
        dim_reduced_X = pca.fit_transform(self.x)
        X = dim_reduced_X

        # plot the input data
        ax = ax or plt.gca()

        centers = pca.fit_transform(self.centers)
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=50, zorder=5, marker='x')

        ax.scatter(X[:, 0], X[:, 1], c=cluster_colors, s=5, cmap='viridis', zorder=2)

        plt.title(self.method_name + ' - ' + self.vectorizer_name + ' (K = {})'.format(self.k))
        plt.show()
