import numpy as np
import xlsxwriter
from sklearn.metrics.cluster import adjusted_rand_score, contingency_matrix, adjusted_mutual_info_score
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



class Cluster:
    def __init__(self, data, vectorizer, model):
        self.data = data
        self.vectorizer = vectorizer(data['text'])
        self.vectors = self.vectorizer.get_vectors(data['text'])
        self.model = model(k=14, vectorizer_name=self.vectorizer.name)
        self.is_clustered = False
        self.fit()

    def fit(self):
        self.model.fit(self.vectors)
        self.is_clustered = True

    def predict(self, raw_docs):
        if not self.is_clustered:
            self.fit()
        vectors = self.vectorizer.get_vectors(raw_docs)
        return self.model.predict(vectors)

    def fit_predict(self):
        if not self.is_clustered:
            self.fit()
        return self.model.predict(self.vectors)

    def adjusted_rand_index_score(self):
        if not self.is_clustered:
            self.fit()
        y_true = self.data['tags']
        # d = dict([(y, x) for x, y in enumerate(sorted(set(Y)))])
        # y_true = [d[x] for x in Y]
        y_pred = self.model.predictions
        return adjusted_rand_score(y_true, y_pred)

    def AMI_score(self):
        if not self.is_clustered:
            self.fit()
        y_true = self.data['tags']
        # d = dict([(y, x) for x, y in enumerate(sorted(set(Y)))])
        # y_true = [d[x] for x in Y]
        y_pred = self.model.predictions
        return adjusted_mutual_info_score(y_true, y_pred)

    def get_purity(self):
        if not self.is_clustered:
            self.fit()

        y_true = self.data['tags']
        # d = dict([(y, x) for x, y in enumerate(sorted(set(self.data['tags'])))])
        # y_true = [d[x] for x in self.data['tags']]
        y_pred = self.model.predictions
        # compute contingency matrix (also called confusion matrix)
        cooccurrence_matrix = contingency_matrix(y_true, y_pred)
        # Y = self.data['tags']
        # d = dict([(y, x) for x, y in enumerate(sorted(set(Y)))])
        # y_true = [d[x] for x in Y]
        # cooccurrence_matrix = self.model.get_cooccurrence_matrix(y_true)
        return sum(cooccurrence_matrix.max(0)) / len(y_true)

    def get_rand_index(self):
        if not self.is_clustered:
            self.fit()

        y_true = self.data['tags']
        # d = dict([(y, x) for x, y in enumerate(sorted(set(self.data['tags'])))])
        # y_true = [d[x] for x in self.data['tags']]
        y_pred = self.model.predictions
        # compute contingency matrix (also called confusion matrix)
        cooccurrence_matrix = contingency_matrix(y_true, y_pred)

        # Y = self.data['tags']
        # d = dict([(y, x) for x, y in enumerate(sorted(set(Y)))])
        # y_true = [d[x] for x in Y]
        # cooccurrence_matrix = self.model.get_cooccurrence_matrix(y_true)

        tp_plus_fp = vComb(cooccurrence_matrix.sum(0)).sum()
        tp_plus_fn = vComb(cooccurrence_matrix.sum(1)).sum()
        tp = vComb(cooccurrence_matrix).sum()
        fp = tp_plus_fp - tp
        fn = tp_plus_fn - tp
        tn = choose2(cooccurrence_matrix.sum()) - tp - fp - fn

        return float(tp + tn) / (tp + fp + fn + tn)

    def report(self):
        write_csv_file('report/' + self.model.method_name + '_' + self.vectorizer.name + '.xlsx',
                       self.data['link'],
                       self.fit_predict())

    def plot_tsne(self):
        data_subset = self.model.x
        y = np.array(self.model.predictions)
        df = pd.DataFrame(data_subset)
        df['y'] = y

        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(data_subset)

        df['tsne-2d-one'] = tsne_results[:, 0]
        df['tsne-2d-two'] = tsne_results[:, 1]
        plt.figure(figsize=(16, len(set(df['y']))))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="y",
            palette=(sns.color_palette("bright") + sns.color_palette("dark") +
                     sns.color_palette("Set3") + sns.color_palette("pastel") +
                     sns.color_palette("muted"))[: len(set(df["y"]))],
            data=df,
            legend="full",
            alpha=0.7
        )
        plt.title('t-sne plot:' + self.model.method_name + ' - ' + self.vectorizer.name +
                  ' (K = {})'.format(self.model.k))
        plt.show()


def choose2(a):
    if a < 2:
        return 0
    return a * (a - 1) / 2


vComb = np.vectorize(choose2)


def write_csv_file(file_path, links, lables):
    workbook = xlsxwriter.Workbook(file_path)
    worksheet = workbook.add_worksheet()
    # Start from the first cell. Rows and columns are zero indexed.
    row = 0
    col = 0

    # Iterate over the data and write it out row by row.
    for i in range(len(links)):
        worksheet.write(row, col, links[i])
        worksheet.write(row, col + 1, lables[i])
        row += 1

    workbook.close()
