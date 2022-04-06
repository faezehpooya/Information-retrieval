import math
import operator
import numpy as np

from Phase2.preprocessor import Preprocessor


class Tf_idf_vectorizer:
    def __init__(self, docs, max_df=0.7, max_features_num=1000):
        self.doc = Preprocessor().preprocess(docs)
        self.N = len(docs)
        terms = self.find_all_terms()
        term_idf = {}
        for t in terms:
            df = self.df(t, self.doc)
            if df > max_df * self.N:
                continue
            idf = math.log(self.N / self.df(t, self.doc), 10)
            term_idf[t] = idf
        top_idf = sorted(term_idf.items(), key=operator.itemgetter(1), reverse=True)
        if len(top_idf) > max_features_num:
            # quantile = int((len(top_idf) - max_features_num)*0.00)
            # quantile = min(3, (len(top_idf) - max_features_num))
            quantile = 0
            top_idf = top_idf[quantile: quantile + max_features_num]
        self.all_terms = [i[0] for i in top_idf]
        self.all_idf = {i[0]: i[1] for i in top_idf}

        # cut = len(term_idf)
        # if cut > max_features_num:
        #     cut = max_features_num
        # samples_index = np.random.choice(len(term_idf), cut, replace=False)
        # samples = np.array(list(term_idf.items()))[samples_index]
        # self.all_terms = [i[0] for i in samples]
        # self.all_idf = {i[0]: float(i[1]) for i in samples}

        print("features num:", len(self.all_terms))

    def vector(self, docsss):
        docss = Preprocessor().preprocess(docsss)
        tf_idf_vector = []
        tf_vector = []
        for d in range(len(docss)):
            vector = []
            tf_V = []
            for t in self.all_terms:
                term_freq = self.tf(t, docss[d])
                tf_V.append(term_freq)
                inverse_doc_freq = self.all_idf.get(t)
                vector.append(term_freq * inverse_doc_freq)
            tf_idf_vector.append(vector)
            tf_vector.append(tf_V)
        return [tf_idf_vector, tf_vector]

    def find_all_terms(self):
        unique_terms = []
        for i in range(len(self.doc)):
            title = self.doc[i]["title"]
            description = self.doc[i]["description"]
            titlewords = title.split()
            descriptionwords = description.split()
            terms = titlewords + descriptionwords
            for term in terms:
                if term not in unique_terms:
                    unique_terms.append(term)
        return unique_terms

    def tf(self, t, d):
        occur = 0
        document = d
        title = document["title"]
        description = document["description"]
        titlewords = title.split()
        descriptionwords = description.split()
        words = titlewords + descriptionwords
        for word in words:
            if t == word:
                occur += 1
        return occur

    def df(self, t, docs):
        occur = 0
        for i in range(len(docs)):
            title = docs[i]["title"]
            description = docs[i]["description"]
            titlewords = title.split()
            descriptionwords = description.split()
            words = titlewords + descriptionwords
            if t in words:
                occur += 1
        return occur
