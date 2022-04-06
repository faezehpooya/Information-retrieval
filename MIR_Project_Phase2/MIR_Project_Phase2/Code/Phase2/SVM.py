from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, \
    f1_score


class SVM:
    def __init__(self, c):
        self.C = c
        self.classifier = LinearSVC(C=c, max_iter=10000)
        self.test_data = np.array(0)
        self.test_tags = np.array(0)
        self.test_predicted_tags = []
        self.train_data = np.array(0)
        self.train_tags = np.array(0)
        self.predicted_tags_for_validation = np.array(0)

    def fit(self, train_data, train_tags):
        self.train_data = np.array(train_data)
        self.train_tags = np.array(train_tags)
        self.classifier.fit(self.train_data, self.train_tags)

    def predict(self, test_data, tags=None):
        self.test_data = np.array(test_data)
        prediction = self.classifier.predict(self.test_data)
        self.test_predicted_tags = prediction
        return prediction

    def _validate_c(self, classifier, predicted_tags: np.array):

        accuracy = accuracy_score(self.test_tags, predicted_tags)
        precision = precision_score(self.test_tags, predicted_tags)
        recall = recall_score(self.test_tags, predicted_tags)
        f1 = f1_score(self.test_tags, predicted_tags)

        print("______________________")
        print()
        print("confusion_matrix: ")
        print(confusion_matrix(self.test_tags, predicted_tags))
        print()
        print("classification_report: ")
        print(classification_report(self.test_tags, predicted_tags))
        print()
        print("accuracy_score: ")
        print(accuracy_score(self.test_tags, predicted_tags))
        print()
        print({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'F1': f1})

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'F1': f1}

    def validate(self, test_data, test_tags):

        self.test_data = np.array(test_data)
        self.test_tags = np.array(test_tags)

        all_validations = dict()
        all_validations['C value' + str(self.C)] = self._validate_c(self.classifier, self.test_predicted_tags)

        for c in [1, 0.5, 1.5, 2]:
            if not c == self.C:
                _svm = LinearSVC(C=c, max_iter=10000)
                _svm.fit(self.train_data, self.train_tags)
                _prediction = _svm.predict(self.test_data)
                all_validations['C value ' + str(c)] = self._validate_c(_svm, _prediction)

        best_accuracy = 0
        best_precision = 0
        best_recall = 0
        best_f1 = 0
        for dic in all_validations.values():

            if dic['accuracy'] > best_accuracy:
                best_accuracy = dic['accuracy']
            if dic['precision'] > best_precision:
                best_precision = dic['precision']
            if dic['recall'] > best_recall:
                best_recall = dic['recall']
            if dic['F1'] > best_f1:
                best_f1 = dic['F1']
        all_validations['best'] = {'accuracy': best_accuracy, 'precision': best_precision, 'recall': best_recall,
                                   'F1': best_f1}
        return all_validations


