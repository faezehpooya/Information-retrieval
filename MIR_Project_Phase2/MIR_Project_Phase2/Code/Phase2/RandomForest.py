from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, \
    f1_score


class RandomForest:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        self.train_data = np.array(0)
        self.train_tags = np.array(0)
        self.test_data = np.array(0)
        self.test_tags = np.array(0)
        self.test_predicted_tags = np.array(0)

    def fit(self, train_data, train_tags):
        for i in range(len(train_tags)):
            if train_tags[i] == -1:
                train_tags[i] = 0
        self.train_data = np.array(train_data)
        self.train_tags = np.array(train_tags)
        self.classifier.fit(self.train_data, self.train_tags)

    def predict(self, test_data, test_tags):
        self.test_data = np.array(test_data)
        self.test_tags = np.array(test_tags)
        for i in range(len(self.test_tags)):
            if self.test_tags[i] == -1:
                self.test_tags[i] = 0
        prediction = self.classifier.predict(self.test_data)
        self.test_predicted_tags = prediction
        prediction_2 = []
        for i in range(len(prediction)):
            if prediction[i] == 1:
                prediction_2.append(prediction[i])
            else:
                prediction_2.append(-1)
        return prediction_2

    def validate(self):
        accuracy = accuracy_score(self.test_tags, self.test_predicted_tags)
        precision = precision_score(self.test_tags, self.test_predicted_tags)
        recall = recall_score(self.test_tags, self.test_predicted_tags)
        f1 = f1_score(self.test_tags, self.test_predicted_tags)

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'F1': f1}
