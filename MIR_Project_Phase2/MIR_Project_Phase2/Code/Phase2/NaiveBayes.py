import math


class NaiveBayes:

    def __init__(self):
        self.conditionals_1 = []
        self.conditionals_minus_1 = []
        self.low_prob = 0
        self.high_prob = 0
        self.original_test_classes = []  # used for validation
        self.predicted_self_classes = []

    def fit(self, vectors, tags, mapping):
        high_view_len = 0
        low_view_len = 0
        for i in range(len(mapping)):
            self.conditionals_minus_1.append(1)
            self.conditionals_1.append(1)
        high_view_len += len(mapping)
        low_view_len += len(mapping)

        for i in range(len(vectors)):
            class_doc = tags[i]
            tf_doc = vectors[i]  # for naive bayes we only use tf
            length_doc = 0
            for t in tf_doc:
                length_doc += t

            if class_doc == 1:
                high_view_len += length_doc
                self.high_prob += 1
            else:
                low_view_len += length_doc
                self.low_prob += 1

            for i in range(len(tf_doc)):
                term_tf = tf_doc[i]
                if class_doc == 1:
                    self.conditionals_1[i] += term_tf
                else:
                    self.conditionals_minus_1[i] += term_tf

        for i in range(len(self.conditionals_1)):
            self.conditionals_1[i] /= high_view_len

        for i in range(len(self.conditionals_minus_1)):
            self.conditionals_minus_1[i] /= low_view_len

        self.low_prob = self.low_prob / len(vectors)
        self.high_prob = self.high_prob / len(vectors)

    def predict(self, vectors, tags):
        for i in range(len(vectors)):
            class_doc = tags[i]
            tf_doc = vectors[i]
            self.original_test_classes.append(class_doc)
            prediction = self._predict_doc(tf_doc)
            self.predicted_self_classes.append(prediction)
        return self.predicted_self_classes

    def _predict_doc(self, doc_list: list):
        prob_high = math.log(self.high_prob)
        prob_low = math.log(self.low_prob)
        for index in range(len(doc_list)):
            p_con_high = self.conditionals_1[index]
            p_con_low = self.conditionals_minus_1[index]
            for i in range(doc_list[index]):
                prob_high += math.log(p_con_high)
                prob_low += math.log(p_con_low)

        if prob_high > prob_low:
            return 1
        else:
            return -1

    def validate(self):
        accuracy = 0
        num_docs = len(self.original_test_classes)
        recall_den = 0
        numerator = 0
        precision_den = 0
        for i in range(num_docs):
            if self.predicted_self_classes[i] == self.original_test_classes[i]:
                if self.predicted_self_classes[i] == 1:
                    numerator += 1
                accuracy += 1
            if self.original_test_classes[i] == 1:
                recall_den += 1
            if self.predicted_self_classes[i] == 1:
                precision_den += 1
        accuracy /= num_docs
        recall = numerator / recall_den
        precision = numerator / precision_den

        f1 = 2 * ((precision * recall) / (precision + recall))
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'F1': f1}






