from sklearn.mixture import GaussianMixture
from Phase3.preprocess import load_processed_data_from_file
from Phase3.Tf_idf_vectorizer import Tf_idf_vectorizer
from Phase3.Word2vec import Word2Vec
import numpy as np

vectorizers = [Tf_idf_vectorizer, Word2Vec]
covariance_types = ['full', 'diag']

GMM_kwargs = {
    "max_iter": 1000,
}

# A list holds the SSE values for each k
scores = [[] for i in range(len(vectorizers))]
max_k = 10
i = 0
data = load_processed_data_from_file()
for covariance_type in covariance_types:
    for vectorizer in vectorizers:
        vectorizer = vectorizer(data['text'])
        data_vectors = vectorizer.get_vectors(data['text'])
        for run in range(10):
            GMM = GaussianMixture(n_components=max_k, covariance_type=covariance_type, **GMM_kwargs)
            GMM.fit(data_vectors)
            scores[i].append(GMM.score(data_vectors, data["tags"]))

        print("the average score of", run, "runs GMM", covariance_type, vectorizer.name,
              "(k = ", max_k + 1, ") is:", np.array(scores[i]).mean())

    i += 1
