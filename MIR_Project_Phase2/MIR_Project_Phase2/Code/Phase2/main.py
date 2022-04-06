from Phase2.Reader import read_file
from Phase2.TF_IDF_VECTORIZE import Tf_idf_vectorizer
from Phase2.TF_IDF_CLASSIFIER import TfIdfClassifier
from Phase2.knn import KNN
from Phase2.NaiveBayes import *
from Phase2.SVM import *
from Phase2.RandomForest import *

a = read_file("data/train.csv")
train_data = a.read_csv_file()

b = read_file("data/test.csv")
test_data = b.read_csv_file()
tfidf_train_vectorizer = Tf_idf_vectorizer(train_data)

model = 'KNN'  # 'naive bayes' or 'SVM' or 'KNN' or 'Random forest'

classifier = TfIdfClassifier(train_data, test_data, tfidf_train_vectorizer, None)

if model == 'KNN':
    print('-> KNN:')
    # knn = TfIdfClassifier(train_data, test_data, tfidf_train_vectorizer, KNN(5))
    classifier.set_model(KNN(5))
    knn = classifier
    knn.fit()
    print("train report:")
    knn.report(train_data)
    print()
    print("test report:")
    knn.report(test_data)
    print("_______________________")
    print()

model = 'naive bayes'
if model == 'naive bayes':
    print('-> Naive Bayes:')
    # naive_bayes = TfIdfClassifier(train_data, test_data, tfidf_train_vectorizer, NaiveBayes())
    classifier.set_model(NaiveBayes())
    naive_bayes = classifier
    naive_bayes.fit()
    print("train report:")
    naive_bayes.report(train_data)
    print()
    print("test report:")
    naive_bayes.report(test_data)
    print("_______________________")
    print()

model = 'SVM'
if model == 'SVM':
    print('-> SVM:')
    # svm = TfIdfClassifier(train_data, test_data, tfidf_train_vectorizer, SVM(1))
    classifier.set_model(SVM(1))
    svm = classifier
    svm.fit()
    print("train report:")
    svm.report(train_data)
    print()
    print("test report:")
    svm.report(test_data)
    print("_______________________")
    print()

model = 'Random forest'
if model == 'Random forest':
    print('-> Random Forest:')
    # rf = TfIdfClassifier(train_data, test_data, tfidf_train_vectorizer, RandomForest())
    classifier.set_model(RandomForest())
    rf = classifier
    rf.fit()
    print("train report:")
    rf.report(train_data)
    print()
    print("test report:")
    rf.report(test_data)
    print("_______________________")
    print()
