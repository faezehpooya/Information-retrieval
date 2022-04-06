from Phase2.Reader import read_file
from Phase2.TF_IDF_VECTORIZE import Tf_idf_vectorizer
from Phase2.TF_IDF_CLASSIFIER import TfIdfClassifier
from Phase2.knn import KNN
from Phase2.NaiveBayes import *
from Phase2.SVM import *
from Phase2.RandomForest import *

a = read_file("../Phase2/data/train.csv")
train_data = a.read_csv_file()

b = read_file("../Phase2/data/test.csv")
test_data = b.read_csv_file()
tfidf_train_vectorizer = Tf_idf_vectorizer(train_data)


def get_best_classifier():
    svm_classifier = TfIdfClassifier(train_data, test_data, tfidf_train_vectorizer, SVM(1))
    svm_classifier.fit()
    return svm_classifier
