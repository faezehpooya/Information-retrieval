from Phase2.TF_IDF_CLASSIFIER import TfIdfClassifier
from Phase2.Reader import read_file
from Phase2.TF_IDF_VECTORIZE import Tf_idf_vectorizer
from Phase2.knn import KNN
from Phase2.SVM import SVM
import numpy as np


def validation(model, parameters_list, validation_ratio=.1):
    """
    :param model: can be "svm" or "knn"
    :param parameters_list: list of parameters
    :param validation_ratio: ratio of data for validation
    :return:
    """
    reader = read_file("data/train.csv")
    train_data = reader.read_csv_file()
    tfidf_vectorizer = Tf_idf_vectorizer(train_data, max_features_num=12000)
    n_samples = len(train_data)
    validation_samples = np.random.choice(n_samples, int(validation_ratio * n_samples), replace=False)
    mask = np.ones(n_samples, dtype=bool)
    mask[validation_samples] = False
    new_train = list(np.array(train_data)[mask])
    validation = list(np.array(train_data)[validation_samples])
    classifier = TfIdfClassifier(new_train, validation, tfidf_vectorizer, None)
    for parameter in parameters_list:
        if model == "SVM":
            new_model = SVM(parameter)
        elif model == 'KNN':
            new_model = KNN(n_neighbours=parameter)

        classifier.set_model(new_model)
        classifier.fit()
        print(" --> parameter: {}".format(parameter))
        print("train report:")
        classifier.report(new_train)
        print("validation report:")
        classifier.report(validation)
        print()


if __name__ == '__main__':
    # SVM
    print("-> SVM")
    validation("SVM", parameters_list=[.5, 1, 1.5, 2])
    print("____________________________________")
    print("-> KNN")
    validation("KNN", parameters_list=[1, 5, 9])
