from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, ward
import seaborn as sns


class HierarchicalClustering:
    def __init__(self, k=3, vectorizer_name=''):
        self.method_name = 'Hierarchical'
        self.vectorizer_name = vectorizer_name
        self.k = k
        self.model = AgglomerativeClustering(n_clusters=self.k, linkage='ward')

    def fit(self, x):
        self.x = x
        self.predictions = self.model.fit_predict(x)

    def predict(self, x):
        return self.model.fit_predict(x)

    def plot(self):
        pca = PCA(n_components=2)
        dim_reduced_data = pca.fit_transform(self.x)
        labels = self.predictions
        colors = (sns.color_palette("bright") + sns.color_palette("dark") +
                  sns.color_palette("Set3") + sns.color_palette("pastel") +
                  sns.color_palette("muted"))
        cluster_colors = [colors[i] for i in labels]
        plt.scatter(dim_reduced_data[:, 0], dim_reduced_data[:, 1],
                    c=cluster_colors, s=5, cmap='viridis', zorder=2)
        plt.title(self.method_name + ' - ' + self.vectorizer_name + ' (K = {})'.format(self.k))
        plt.show()
        self.plot_dendrogram()

    def plot_dendrogram(self):
        linkage_matrix = ward(self.x)
        dendrogram(linkage_matrix, truncate_mode='level', p=5)
        plt.title('Dendrogram: ' + self.method_name + ' - ' + self.vectorizer_name +
                  ' (K = {})'.format(self.k))
        plt.show()
