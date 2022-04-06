from sklearn.feature_extraction.text import TfidfVectorizer


class Tf_idf_vectorizer:
    def __init__(self, raw_docs):
        self.name = 'tfidf'
        self.vectorizer = TfidfVectorizer(max_df=0.85, max_features=1000)
        self.vectorizer.fit(raw_docs)

    def get_vectors(self, raw_docs):
        return self.vectorizer.transform(raw_docs).toarray()
