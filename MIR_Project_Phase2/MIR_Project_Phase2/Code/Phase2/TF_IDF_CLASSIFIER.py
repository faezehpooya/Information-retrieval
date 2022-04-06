from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from Phase2.NaiveBayes import *
from Phase2.knn import *
from Phase2.SVM import *
from Phase2.RandomForest import *


class TfIdfClassifier:
    """
        train_data: tuple(train_docs, train_tags)
        test_data: tuple(test_docs, test_tags)
    """

    def __init__(self, train_data, test_data, tfidf_train_vectorizer, model):
        self.tf_idf_vectorizer = tfidf_train_vectorizer
        self.train_docs_sparse_matrix = self.tf_idf_vectorizer.vector(train_data)[0]
        self.train_docs_sparse_matrix_tf = self.tf_idf_vectorizer.vector(train_data)[1]
        self.train_tags = [sub["view"] for sub in train_data]
        self.test_docs_sparse_matrix = self.tf_idf_vectorizer.vector(test_data)[0]
        self.test_docs_sparse_matrix_tf = self.tf_idf_vectorizer.vector(test_data)[1]
        self.test_tags = [sub["view"] for sub in test_data]
        self.model = model
        self.is_learned = False

    def set_model(self, model):
        self.model = model

    def fit(self):
        if type(self.model) == NaiveBayes:
            self.model.fit(self.train_docs_sparse_matrix_tf, self.train_tags, self.tf_idf_vectorizer.all_terms)
            self.is_learned = True

        if type(self.model) == KNN:
            self.model.fit(self.train_docs_sparse_matrix, self.train_tags)
            self.is_learned = True
        if type(self.model) == SVM:
            self.model.fit(self.train_docs_sparse_matrix, self.train_tags)
            self.is_learned = True
        if type(self.model) == RandomForest:
            self.model.fit(self.train_docs_sparse_matrix, self.train_tags)
            self.is_learned = True

    def predict(self, raw_doc, tags=None):
        if not self.is_learned:
            self.fit()

        test_tags = self.test_tags
        if tags:
            test_tags = tags

        if type(self.model) == NaiveBayes:
            return self.model.predict(self.tf_idf_vectorizer.vector(raw_doc)[1], test_tags)

        if type(self.model) == KNN:
            doc_sparse_matrix = self.tf_idf_vectorizer.vector(raw_doc)[0]
            return self.model.predict(doc_sparse_matrix)

        if type(self.model) == SVM:
            return self.model.predict(self.tf_idf_vectorizer.vector(raw_doc)[0])

        if type(self.model) == RandomForest:
            return self.model.predict(self.tf_idf_vectorizer.vector(raw_doc)[0], self.test_tags)

    """
        raw_data: tuple(taw_docs, tags)
        if raw_data is None get report of test_data
    """

    def report(self, raw_test_data):
        if raw_test_data:
            tags = [sub["view"] for sub in raw_test_data]
        else:
            tags = self.test_tags
        predicted_tags = self.predict(raw_test_data, tags)
        accuracy = 0
        num_docs = len(tags)
        recall_den = 0
        numerator = 0
        precision_den = 0
        for i in range(num_docs):
            if predicted_tags[i] == tags[i]:
                if predicted_tags[i] == 1:
                    numerator += 1
                accuracy += 1
            if tags[i] == 1:
                recall_den += 1
            if predicted_tags[i] == 1:
                precision_den += 1
        accuracy /= num_docs
        recall = numerator
        if recall_den:
            recall /= recall_den
        precision = numerator
        if precision_den:
            precision /= precision_den

        f1 = (precision * recall)
        if precision + recall:
            f1 /= (precision + recall)
        f1 *= 2

        # print("______________________")
        # print()
        # print("confusion_matrix: ")
        # print(confusion_matrix(tags, predicted_tags))
        # print()
        # print("classification_report: ")
        # print(classification_report(tags, predicted_tags))
        # print()
        # print("accuracy_score: ")
        # print(accuracy_score(tags, predicted_tags))
        # print()

        print('accuracy:', accuracy, 'precision:', precision, 'recall:', recall, 'F1:', f1)

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'F1': f1}

    def validate(self):
        if type(self.model) == NaiveBayes:
            return self.model.validate()
        if type(self.model) == SVM:
            return self.model.validate(self.test_docs_sparse_matrix, self.test_tags)
        if type(self.model) == RandomForest:
            return self.model.validate()
