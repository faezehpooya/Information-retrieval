import os, pickle
from Phase1.Index.Comperss.GammaCode import GammaCodeCompressor, GammaCodeDecompressor
from Phase1.Index.Comperss.VariableByte import VariableByteCompressor, VariableByteDecompressor


class Indexer:
    def __init__(self, docs_directory='all_docs/', index_file='index.pkl'):

        self.all_docs = []  # list of doc_ids
        self.dict_terms = {}  # {term: {'t_id', 'df'}}
        self.inv_term_mapping = {}  # {t_id: term}
        self.positional_index = {}  # positional indexing {t_id: {doc_id: {'title': [pos], 'description': [pos]}}}
        self.bigram_index = {}  # bigram indexing {bigram: {t_id}}

        self.generate_id = 0

        self.docs_directory = docs_directory
        self.index_file = index_file

    def save_index(self, file_name="index.pkl"):
        pickle.dump(self, open(self.index_file, 'wb'), pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(index_file='index.pkl'):
        return pickle.load(open(index_file, 'rb'))

    def save_posting(self, compression_type="NotCompressed", file_name=None):
        if file_name is None:
            file_name = compression_type + "_index.txt"
        if compression_type == "NotCompressed":
            with open(file_name, 'w+') as file:
                file.write(str(self.positional_index))
                file.close()
        elif compression_type == 'GammaCode':
            GammaCodeCompressor.write_compress_to_file(self.positional_index, file_name)
        elif compression_type == 'VariableByte':
            VariableByteCompressor.write_compress_to_file(self.positional_index, file_name)

    def load_posting(self, compression_type="NotCompressed", file_name=None):
        if file_name is None:
            file_name = compression_type + "_index.txt"
        if compression_type == "NotCompressed":
            with open(file_name, 'r') as file:
                self.positional_index = eval(file.read())
                file.close()
        elif compression_type == 'GammaCode':
            self.positional_index = GammaCodeDecompressor.decompress_from_file(file_name)
        elif compression_type == 'VariableByte':
            self.positional_index = VariableByteDecompressor.decompress_from_file(file_name)

    def save_bigram_posting(self, file_name="Bigram_index.txt"):
        with open(file_name, 'w+') as file:
            file.write(str(str(self.bigram_index).encode('utf8')))
            file.close()

    def load_bigram_posting(self, file_name="Bigram_index.txt"):
        with open(file_name, 'r') as file:
            self.bigram_index = eval(eval(file.read()).decode('utf8'))
            file.close()

    def load_doc(self, doc_id):
        file = open(self.docs_directory + str(doc_id) + '.txt', 'r')
        doc = eval(file.read())
        doc['title'] = doc['title'].decode('utf8')
        doc['description'] = doc['description'].decode('utf8')
        return doc

    def add_doc(self, doc, doc_id):
        write_doc = {}
        write_doc['title'] = doc['title'].encode('utf8')
        write_doc['description'] = doc['description'].encode('utf8')
        with open(self.docs_directory + str(doc_id) + '.txt', 'w+') as file:
            file.write(str(write_doc))
            file.close()
        self.all_docs.append(doc_id)

        self.add_doc_to_index(doc_id)

    def del_doc(self, doc_id):
        self.remove_doc_from_index(doc_id)

        os.remove(self.doc_directory + str(doc_id) + '.txt')
        self.all_docs.remove(doc_id)

    def add_terms(self, doc_id, terms, header):
        for i in range(len(terms)):
            term = terms[i]
            if self.dict_terms.get(term) is None:
                term_id = self.generate_id
                self.generate_id += 1
                self.dict_terms[term] = {'t_id': term_id, 'df': 0}
                self.inv_term_mapping[term_id] = term
                self.add_term_to_bigram_index(term)
            term_id = self.dict_terms[term]['t_id']

            posting = self.positional_index.get(term_id)
            if posting is None:
                self.positional_index[term_id] = {}
                posting = self.positional_index.get(term_id)

            doc_posting = posting.get(doc_id)
            if doc_posting is None:
                self.dict_terms[term]['df'] += 1

                posting[doc_id] = {}
                doc_posting = posting.get(doc_id)

            positions = doc_posting.get(header)
            if positions is None:
                doc_posting[header] = []
                positions = doc_posting.get(header)
            positions.append(i)

    def add_doc_to_index(self, doc_id):
        doc = self.load_doc(doc_id)

        title_terms = doc['title'].split()
        self.add_terms(doc_id, title_terms, 'title')

        description_terms = doc['description'].split()
        self.add_terms(doc_id, description_terms, 'description')

    def remove_doc_from_index(self, doc_id):
        removed_terms = []
        for t_id in self.positional_index:
            self.positional_index.get(t_id).pop(doc_id, None)
            self.dict_terms[self.inv_term_mapping.get(t_id)]['df'] -= 1
            if self.positional_index.get(t_id) == {}:
                removed_terms.append(t_id)
        for t_id in removed_terms:
            self.positional_index.pop(t_id, None)
            term = self.inv_term_mapping.pop(t_id, None)
            self.remove_term_from_bigram_index(term)
            self.dict_terms.pop(term, None)

    def add_term_to_bigram_index(self, term):
        t_id = self.dict_terms.get(term)['t_id']
        term = '$' + term + '$'
        for i in range(len(term) - 1):
            bigram = term[i:(i + 2)]
            bigrams_t_ids = self.bigram_index.get(bigram)
            if bigrams_t_ids is None:
                self.bigram_index[bigram] = set()
                bigrams_t_ids = self.bigram_index.get(bigram)
            bigrams_t_ids.add(t_id)

    def remove_term_from_bigram_index(self, term):
        t_id = self.dict_terms.get(term)['t_id']
        term = '$' + term + '$'
        for i in range(len(term) - 1):
            bigram = term[i:(i + 2)]
            bigrams_t_ids = self.bigram_index.get(bigram)
            if t_id in bigrams_t_ids:
                bigrams_t_ids.remove(t_id)

    def get_posting(self, term):
        term_dict = self.dict_terms.get(term)
        if term_dict:
            t_id = term_dict.get('t_id')
            return list(self.positional_index.get(t_id).keys())
        else:
            return "\"" + term + "\" does not exist."

    def get_posting_with_positions(self, term):
        term_dict = self.dict_terms.get(term)
        if term_dict:
            t_id = term_dict.get('t_id')
            return self.positional_index.get(t_id)
        else:
            return "\"" + term + "\" does not exist."

    def get_bigram_posting(self, bigram):
        bigrams_t_ids = self.bigram_index.get(bigram)
        terms = []
        if bigrams_t_ids is not None:
            for t_id in bigrams_t_ids:
                term = self.inv_term_mapping.get(t_id)
                if term is not None:
                    terms.append(term)
        return terms

    def get_tf(self, term, doc_id):
        tf = {'title': 0, 'description': 0}
        if not self.dict_terms.get(term):
            return tf
        t_id = self.dict_terms[term]['t_id']
        posting_list = self.positional_index.get(t_id)
        if not posting_list.get(doc_id):
            return tf
        doc_posting = posting_list[doc_id]
        if doc_posting.get('title'):
            tf['title'] = len(doc_posting['title'])
        if doc_posting.get('description'):
            tf['description'] = len(doc_posting['description'])
        return tf

    def get_df(self, term):
        if not self.dict_terms.get(term):
            return 0
        return self.dict_terms[term]['df']

    def get_number_of_docs(self):
        return len(self.all_docs)

    def tf_vector(self, terms, doc_id):
        vector = {'title': [], 'description': []}
        for term in terms:
            tf = self.get_tf(term, doc_id)
            title_tf = tf['title']
            description_tf = tf['description']
            vector['title'].append(title_tf)
            vector['description'].append(description_tf)
        return vector

    def tf_vector_of_all_docs(self, terms):
        tf_vectors = {}
        for doc_id in self.all_docs:
            tf_vectors[doc_id] = self.tf_vector(terms, doc_id)
        return tf_vectors
