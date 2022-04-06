import json
import os
import pickle
import re
from abc import abstractmethod

import hazm
from hazm import Stemmer


class GenericPreprocessor:

    def __init__(self):
        self.processed_list = None
        self.high_accur_param = None
        self.stop_words = None
        self.high_accured_words = None

    def preprocess(self, text_list, is_query=False):
        """
        :param text_list: ['text']
        :return:
        """
        self.processed_list = []
        normalized_list = []
        if not is_query:
            for news in text_list:
                self.processed_list.append(
                    {"title": self.normalize(news["title"]), "summary": self.normalize(news["summary"]),
                     "link": news["link"], "tags": (news["tags"])})

            self.set_stopwords()
            self.remove_stopwords()

            for news in self.processed_list:
                title = news["title"]
                summary = news["summary"]
                link = news["link"]
                tags = news["tags"]
                title_stem = self.__stem_doc(title)
                summary_stem = self.__stem_doc(summary)
                link_stem = self.__stem_doc(link)
                news["text"] = title_stem + summary_stem
                # news["summary"] = summary_stem
                news["link"] = link_stem
                news["tags"] = tags
                normalized_list.append(news)
            self.processed_list = normalized_list

        else:
            processed_list = []
            for news in text_list:
                processed_list.append(self.normalize(news))

            #             self.set_stopwords()
            #             self.remove_stopwords()

            for news in processed_list:
                text = self.__stem_doc(news)
                if self.stop_words:
                    for word in text.split():
                        if word not in self.stop_words:
                            normalized_list.append(word)
                else:
                    normalized_list.append(text)
            processed_list = normalized_list
            normalized_list = ' '.join(normalized_list)

        return normalized_list

    def __stem_doc(self, doc):
        normalized_words = []
        for word in self.__get_word_by_word(doc):
            nword = self.stem(word)
            if nword is not None and nword != '':
                normalized_words.append(nword)
        return ' '.join(normalized_words)

    def __get_word_by_word(self, doc_str):
        words = self.tokenize(doc_str)
        for word in words:
            yield word

    @abstractmethod
    def tokenize(self, doc_str):
        pass

    @abstractmethod
    def normalize(self, text):
        pass

    @abstractmethod
    def stem(self, word):
        pass

    def set_stopwords(self):
        self.high_accured_words = self.__find_high_accured_words()
        self.stop_words = set()
        for key, value in self.high_accured_words.items():
            self.stop_words.add(key)

    def remove_punctuation(self, word):
        return re.sub(r'[^\w\s]', '', word)

    def __get_accurance_dict(self):
        accurance_dict = {}
        for news in self.processed_list:
            title = news["title"]
            summary = news["summary"]
            link = news["link"]
            tags = news["tags"]
            titlewords = title.split()
            summarywords = summary.split()
            words = titlewords + summarywords
            for word in words:
                accurance_dict[word] = accurance_dict.get(word, 0) + 1
        return accurance_dict

    def __find_high_accured_words(self):
        accurance_dict = self.__get_accurance_dict()
        accurance_dict = dict(reversed((sorted(accurance_dict.items(), key=lambda x: x[1]))))
        high_accured_words = dict()
        for key, value in accurance_dict.items():
            if value >= self.high_accur_param:
                high_accured_words[key] = value
        return high_accured_words

    def get_high_accured_words(self):
        return dict(reversed((sorted(self.high_accured_words.items(), key=lambda x: x[1]))))

    def remove_stopwords(self):
        updated_processed_list = []
        for news in self.processed_list:
            title = news["title"]
            summary = news["summary"]
            link = news["link"]
            tags = news["tags"]
            titlewords = title.split()
            summarywords = summary.split()
            updated_news = ""
            for word in titlewords:
                if word not in self.stop_words:
                    updated_news += word + " "
            news["title"] = updated_news
            updated_news = ""
            for word in summarywords:
                if word not in self.stop_words:
                    updated_news += word + " "
            news["summary"] = updated_news
            updated_processed_list.append(news)
        self.processed_list = updated_processed_list
        return self.processed_list


class PersianPreprocessor(GenericPreprocessor):

    def __init__(self):
        super().__init__()
        self.stemmer = Stemmer()
        self.high_accur_param = 3500

    def tokenize(self, doc_str):
        return hazm.word_tokenize(doc_str)

    def normalize(self, text):
        text = self.remove_punctuation(text)
        text = re.sub(' +', ' ', text.strip())
        return text

    def stem(self, word):
        word = self.stemmer.stem(word)
        word = re.sub(' +', ' ', word.strip())
        return word


def read_and_process(file_name='data/hamshahri.json'):
    f = open(file=file_name, encoding="utf8")

    dat = json.load(f)
    data = dat
    pp = PersianPreprocessor()
    per_docs = pp.preprocess(data)
    data = {'text': [], 'tags': [], 'link': []}
    for doc in per_docs:
        data['text'].append(doc["text"])
        data['tags'].append(doc["tags"][0].split('>')[0])
        data['link'].append(doc["link"])
    f.close()
    save_processed_data_to_file(data)
    # print(data["tags"][0].split('>')[0])
    return data


def save_processed_data_to_file(data, file_name='cache/processed_data'):
    pickle.dump(data, open(file_name, mode='wb'), pickle.HIGHEST_PROTOCOL)


def load_processed_data_from_file(data_file='data/hamshahri.json', saved_file_name='cache/processed_data'):
    if not os.path.exists(saved_file_name):
        return read_and_process(data_file)
    return pickle.load(open(saved_file_name, mode='rb'))


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.colors as color
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from Phase3.Word2vec import Word2Vec
from Phase3.Tf_idf_vectorizer import Tf_idf_vectorizer
import seaborn as sns
import time


def plot(data):
    X = Tf_idf_vectorizer(data['text']).get_vectors(data['text'])
    y = np.array(data['tags'])
    feat_cols = ['pixel' + str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    X, y = None, None
    print('Size of the dataframe: {}'.format(df.shape))

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[feat_cols].values)
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    df['pca-three'] = pca_result[:, 2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    plt.figure(figsize=(16, len(set(df['y']))))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=(sns.color_palette("bright") + sns.color_palette("dark") +
                 sns.color_palette("Set3") + sns.color_palette("pastel") +
                 sns.color_palette("muted"))[: len(set(df["y"]))],
        data=df,
        legend="full",
        alpha=0.7
    )
    plt.show()

    ax = plt.figure(figsize=(16,  len(set(df['y'])))).gca(projection='3d')
    ax.scatter(
        xs=df["pca-one"],
        ys=df["pca-two"],
        zs=df["pca-three"],
        c=sns.color_palette("hls", len(df['y'])),
        cmap='tab10'
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    plt.show()

    df_subset = df.copy()
    data_subset = df_subset[feat_cols].values
    # pca = PCA(n_components=3)
    # pca_result = pca.fit_transform(data_subset)
    # df_subset['pca-one'] = pca_result[:, 0]
    # df_subset['pca-two'] = pca_result[:, 1]
    # df_subset['pca-three'] = pca_result[:, 2]

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, len(set(df['y']))))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=(sns.color_palette("bright") + sns.color_palette("dark") +
                 sns.color_palette("Set3") + sns.color_palette("pastel") +
                 sns.color_palette("muted"))[: len(set(df_subset["y"]))],
        data=df_subset,
        legend="full",
        alpha=0.7
    )
    plt.show()

    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(data_subset)

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(pca_result_50)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    df_subset['tsne-pca50-one'] = tsne_pca_results[:, 0]
    df_subset['tsne-pca50-two'] = tsne_pca_results[:, 1]
    plt.figure(figsize=(16, 100))
    sns.scatterplot(
        x="tsne-pca50-one", y="tsne-pca50-two",
        hue="y",
        palette=(sns.color_palette("bright") + sns.color_palette("dark") +
                 sns.color_palette("Set3") + sns.color_palette("pastel") +
                 sns.color_palette("muted"))[: len(set(df_subset["y"]))],
        data=df_subset,
        legend="full",
        alpha=0.7,
    )

    plt.show()


# plot(read_and_process())
