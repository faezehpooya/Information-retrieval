import csv
import xml.etree.ElementTree as ET
import re
from abc import abstractmethod
import unicodedata
import nltk
from nltk.stem import PorterStemmer
import hazm
from hazm import Stemmer


# nltk.download('punkt')


class read_file:
    def __init__(self, csv_file, persion_file):
        self.csv_file = csv_file
        self.persion_file = persion_file

    def get_doc(self, doc_id, language='English'):
        if language == 'persian':
            tree = ET.parse(self.persion_file)
            root = list(tree.getroot())
            prefix_element_name = "{http://www.mediawiki.org/xml/export-0.10/}"

            page = root[doc_id]
            title = page.find(prefix_element_name + 'title').text
            title = re.sub(' +', ' ', title.strip())
            text = page.find(prefix_element_name + 'revision').find(prefix_element_name + 'text').text
            text = re.sub(' +', ' ', text.strip())
            doc = dict()
            doc["title"] = title
            doc["description"] = text

        else:
            file_path = self.csv_file
            with open(file_path, mode='r', encoding="utf8") as csv_file:
                csv_reader = list(csv.DictReader(csv_file))
                row = csv_reader[doc_id]
                title = re.sub(' +', ' ', row["title"].strip())
                description = re.sub(' +', ' ', row["description"].strip())
                views = re.sub(' +', ' ', row["views"].strip())
                doc = dict()
                doc["title"] = title
                doc["description"] = description
                v = -1
                if int(views) > 1698297:
                    v = 1
                doc["views"] = v

        return doc

    def read_csv_file_as_list(self):
        file_path = self.csv_file
        eng_list = []
        with open(file_path, mode='r', encoding="utf8") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                title = re.sub(' +', ' ', row["title"].strip())
                description = re.sub(' +', ' ', row["description"].strip())
                d = dict()
                d["title"] = title
                d["description"] = description
                eng_list.append(d)
            return eng_list

    def read_persian_xml_file_as_list(self):
        tree = ET.parse(self.persion_file)
        root = tree.getroot()
        prefix_element_name = "{http://www.mediawiki.org/xml/export-0.10/}"
        per_list = []
        for page in root:
            title = page.find(prefix_element_name + 'title').text
            title = re.sub(' +', ' ', title.strip())
            text = page.find(prefix_element_name + 'revision').find(prefix_element_name + 'text').text
            text = re.sub(' +', ' ', text.strip())
            d = dict()
            d["title"] = title
            d["description"] = text
            per_list.append(d)
        return per_list


class GenericPreprocessor:

    def __init__(self):
        self.processed_list = None
        self.high_accur_param = None
        self.stop_words = None
        self.high_accured_words = None

    def preprocess(self, text_list, is_query=False):
        """
        :param is_query:
        :param text_list: ['text']
        :return:
        """
        self.processed_list = []
        normalized_list = []
        if not is_query:
            for news in text_list:
                self.processed_list.append(
                    {"title": self.normalize(news["title"]), "description": self.normalize(news["description"])})

            self.set_stopwords()
            self.remove_stopwords()

            for news in self.processed_list:
                title = news["title"]
                description = news["description"]
                title_stem = self.__stem_doc(title)
                description_stem = self.__stem_doc(description)
                news["title"] = title_stem
                news["description"] = description_stem
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
            description = news["description"]
            titlewords = title.split()
            descriptionwords = description.split()
            words = titlewords + descriptionwords
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
            updated_news = ""
            title = news["title"]
            description = news["description"]
            titlewords = title.split()
            descriptionwords = description.split()
            for word in titlewords:
                if word not in self.stop_words:
                    updated_news += word + " "
            news["title"] = updated_news
            updated_news = ""
            for word in descriptionwords:
                if word not in self.stop_words:
                    updated_news += word + " "
            news["description"] = updated_news
            updated_processed_list.append(news)
        self.processed_list = updated_processed_list
        return self.processed_list


class EnglishPreprocessor(GenericPreprocessor):

    def __init__(self):
        super().__init__()
        self.stemmer = PorterStemmer()
        self.high_accur_param = 1300

    def tokenize(self, doc_str):
        return nltk.word_tokenize(doc_str)

    def normalize(self, text):
        text = self.remove_non_ascii(text)
        text = self.remove_punctuation(text)
        text = self.lower(text)
        text = re.sub(' +', ' ', text.strip())
        return text

    def stem(self, word):
        word = self.stemmer.stem(word)
        word = re.sub(' +', ' ', word.strip())
        return word

    @staticmethod
    def remove_non_ascii(word):
        return unicodedata.normalize('NFKD', word) \
            .encode('ascii', 'ignore').decode('utf-8', 'ignore')

    @staticmethod
    def lower(word):
        return word.lower()


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
