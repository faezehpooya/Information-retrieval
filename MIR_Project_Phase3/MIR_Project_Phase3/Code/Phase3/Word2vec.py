from gensim.sklearn_api import W2VTransformer
from gensim.utils import simple_preprocess
import numpy as np


class Word2Vec:

    def __init__(self, raw_docs):
        self.name = 'word2vec'
        self.train_docs = [simple_preprocess(doc) for doc in raw_docs]
        self.dim = 100
        self.vectorizer = W2VTransformer(size=self.dim,
                                         alpha=0.025,
                                         window=5,
                                         iter=20)
        self.vectorizer.fit(self.train_docs)

    def get_vectors(self, raw_docs):
        docs = [simple_preprocess(doc) for doc in raw_docs]
        vectors = []
        for doc in docs:
            doc_words_vector_list = []
            for word in doc:
                try:
                    doc_words_vector_list.append(self.vectorizer.transform(word))
                except:
                    continue  # word is not in vocabulary
            if len(doc_words_vector_list) > 0:
                dec_vector = np.mean(doc_words_vector_list, axis=0)
            else:
                dec_vector = np.zeros((1, self.dim))
            vectors.append(dec_vector)
        vectors = np.concatenate(vectors, axis=0)
        return vectors