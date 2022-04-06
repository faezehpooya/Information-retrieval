import numpy as np
from indexer import *
import operator
import heapq
import math


class Searcher:
    def __init__(self, ind):
        self.indexer = ind
        self.K = 10
        self.tf_ratio = 0.5

    def query_weight(self, query: list):
        term_frequency = {}

        for term in query:
            if term not in term_frequency.keys():
                term_frequency[term] = 1
            else:
                term_frequency[term] += 1

        query_vector = {}
        #       l
        for term in term_frequency.keys():
            query_vector[term] = 1 + np.log(term_frequency[term])
        #       t
        for term in term_frequency.keys():
            N = len(self.indexer.all_docs)
            query_vector[term] *= (np.log(N) - np.log(self.indexer.get_df(term)))
        #       c
        summation = 0
        for term in term_frequency.keys():
            summation += query_vector[term] * query_vector[term]
        summation = math.sqrt(summation)

        for term in term_frequency.keys():
            query_vector[term] /= summation

        return query_vector

    def doc_weight_parametric(self, doc_ids: list, query_vector: dict, parameter):
        doc_weight_dict = {}  # {doc_id: weight}

        for doc_id in doc_ids:
            doc_vector = {}
            for term in query_vector.keys():
                # return value of get_tf is a dict: {'title': tf in title, 'description': tf in desvription}
                tf_dic = self.indexer.get_tf(term, doc_id)
                tf = tf_dic[parameter]
                l_term = 0
                if tf > 0:
                    l_term = 1 + np.log(tf)
                doc_vector[term] = l_term
            summation = 0
            for l in doc_vector.values():
                summation += l * l
            summation = math.sqrt(summation)
            if summation != 0:
                for v in doc_vector.keys():
                    doc_vector[v] /= summation

            doc_weight_dict[doc_id] = doc_vector

        return doc_weight_dict  # {doc_id: {term_id: weight}}

    def doc_weight(self, doc_ids: list, query_vector: dict, parameter=None):

        doc_weight_dict_pre = {}
        doc_weight_dict = {}  # {doc_id: weight}

        # for doc_id in doc_ids:
        # doc_vector = {}
        # for term in query_vector.keys():
        #     # return value of get_tf is a dict: {'title': tf in title, 'description': tf in desvription}
        #     tf_dic = self.indexer.get_tf(term, doc_id)
        #     #tf = self.indexer.get_tf(term, doc_id)
        #     tf_title = tf_dic['title']
        #     tf_des = tf_dic['description']
        #     tf =
        #     l_term = 0
        #     if tf > 0:
        #         l_term = 1 + np.log()
        #     doc_vector[term] = l_term
        # summation = 0
        # for l in doc_vector.values():
        #     summation += l * l
        # summation = math.sqrt(summation)
        # for v in doc_vector.keys():
        #     doc_vector[v] /= summation

        # weight = 0
        # for term in doc_vector.keys():
        #     weight += query_vector[term] + doc_vector[term]
        # doc_weight_dict[doc_id] = weight
        if parameter:
            doc_weight_dict_pre = self.doc_weight_parametric(doc_ids, query_vector,
                                                             parameter)  # {doc_id: {term_id: weight}}
        else:
            doc_weight_dict_title = self.doc_weight_parametric(doc_ids, query_vector, 'title')
            doc_weight_dict_des = self.doc_weight_parametric(doc_ids, query_vector, 'description')
            doc_weight_dict_pre = {}
            for doc_id in doc_weight_dict_des.keys():
                doc_dic = {}
                for term_id in doc_weight_dict_des[doc_id]:
                    weighted_weight = self.tf_ratio * doc_weight_dict_des[doc_id][term_id] + \
                                      (1 - self.tf_ratio) * doc_weight_dict_title[doc_id][term_id]
                    doc_dic[term_id] = weighted_weight
                doc_weight_dict_pre[doc_id] = doc_dic

        for doc_id in doc_weight_dict_pre.keys():
            weight = 0
            for term_id in doc_weight_dict_pre[doc_id].keys():
                weight += query_vector[term_id] * doc_weight_dict_pre[doc_id][term_id]
            doc_weight_dict[doc_id] = weight

        # doc_weight_dict : {doc_id: weight}

        top_docs = sorted(doc_weight_dict.items(), key=operator.itemgetter(1), reverse=True)
        if len(top_docs) > self.K:
            top_docs = top_docs[0: self.K]
        top_doc_ids = [i[0] for i in top_docs]
        return top_doc_ids

    def search(self, query: str, parameter=None):
        query = query.split()
        positional_index = self.indexer.positional_index

        query_vector = self.query_weight(query)
        term_ids = []
        for term in query_vector.keys():
            term_ids.append(self.indexer.dict_terms[term]['t_id'])

        related_doc_ids_list = set()
        for term_id in term_ids:
            if term_id in positional_index.keys():
                docs_related = positional_index[term_id]

                if not parameter:
                    for doc_id in docs_related.keys():
                        related_doc_ids_list = set.union(
                            related_doc_ids_list,
                            set([doc_id])
                        )
                else:
                    for doc_id, v in docs_related.items():
                        if v.get(parameter) and not len(v[parameter]) == 0:
                            related_doc_ids_list = set.union(
                                related_doc_ids_list,
                                set([doc_id])
                            )  # todo: intersection?

        related_doc_ids_list = list(related_doc_ids_list)
        return self.doc_weight(related_doc_ids_list, query_vector)

    def proximity_search(self, query: str, window_size: int, parameter=None):
        query = query.split()
        if len(query) <= 0:
            return []
        positional_index = self.indexer.positional_index
        related_doc_ids_list = set(self.indexer.all_docs)

        #         term = query[0]
        #         term_id = self.indexer.dict_terms[term]['t_id']
        #         if term_id in positional_index:
        #             docs_related = positional_index[term_id]
        #             related_doc_ids_list = set(list(docs_related.keys()))

        for term in query:
            term_id = self.indexer.dict_terms[term]['t_id']
            if term_id in positional_index:
                docs_related = positional_index[term_id]

                related_doc_ids_list = set.intersection(
                    related_doc_ids_list,
                    set(list(docs_related.keys()))
                )

        query_vector = self.query_weight(query)
        related_doc_ids_list = list(related_doc_ids_list)

        term_ids = []
        for term in query_vector.keys():
            term_ids.append(self.indexer.dict_terms[term]['t_id'])

        if parameter:
            proximity_docs = self.proximity_parametric(term_ids, parameter, related_doc_ids_list, window_size)
        else:
            proximity_docs_des = self.proximity_parametric(term_ids, 'description', related_doc_ids_list, window_size)
            proximity_docs_title = self.proximity_parametric(term_ids, 'title', related_doc_ids_list, window_size)
            proximity_docs = list(set.union(set(proximity_docs_title), set(proximity_docs_des)))

        return self.doc_weight(proximity_docs, query_vector)

    def proximity_parametric(self, term_ids, parameter, related_doc_ids_list, window_size):
        proximity_docs = []
        for doc_id in related_doc_ids_list:
            all_postings = {}
            heap = []

            rel = True
            for term_id in term_ids:
                title_and_des = self.indexer.positional_index[term_id][doc_id]
                if title_and_des.get(parameter):
                    all_postings[term_id] = title_and_des[parameter]
                else:
                    rel = False
                    break
            if not rel:
                continue

            for term_id, posting_list in all_postings.items():
                index_in_posting = 0
                heapq.heappush(heap, (posting_list[index_in_posting], term_id, index_in_posting))

            while True:
                if (heapq.nlargest(1, heap)[0][0] - heapq.nsmallest(1, heap)[0][0]) < window_size:
                    proximity_docs.append(doc_id)
                    break
                (p, term_id, index_in_posting) = heapq.heappop(heap)
                posting_list = all_postings[term_id]
                if index_in_posting == len(posting_list) - 1:
                    break
                new_item = (posting_list[index_in_posting + 1], term_id, index_in_posting + 1)
                heapq.heappush(heap, new_item)

        return proximity_docs

