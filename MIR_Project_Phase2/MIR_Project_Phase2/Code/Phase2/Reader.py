import csv
import xml.etree.ElementTree as ET
import re
import numpy as np


class read_file:
    def __init__(self, csv_file, size=None):
        self.csv_file = csv_file
        self.size = size

    def read_csv_file(self):
        file_path = self.csv_file
        text_list = []
        tag = []
        with open(file_path, mode='r', encoding="utf8") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                line_count += 1
                title = re.sub(' +', ' ', row["title"].strip())
                description = re.sub(' +', ' ', row["description"].strip())
                d = dict()
                d["title"] = title
                d["description"] = description
                d["view"] = int(row["views"])
                text_list.append(d)
            n_samples = len(text_list)
            cut = n_samples
            if self.size:
                cut = self.size
            train_samples = np.random.choice(n_samples, cut, replace=False)
            new_train = list(np.array(text_list)[train_samples])
            # print(train_samples)
            return new_train
