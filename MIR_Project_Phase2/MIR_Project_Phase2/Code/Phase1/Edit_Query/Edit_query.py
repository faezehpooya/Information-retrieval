class EditQuery:  # todo: two first parts need rewrite

    def __init__(self, query, indexer, eng_preprocessor, persian_preprocessor):
        self.eng_preprocessor = eng_preprocessor
        self.persian_preprocessor = persian_preprocessor
        self.indexer = indexer
        is_eng = EditQuery.is_english(query)
        if is_eng:
            normalized_query = eng_preprocessor.preprocess([query], is_query=True)
        else:
            normalized_query = persian_preprocessor.preprocess([query], is_query=True)
        self.query_token_list = normalized_query.split()

    @staticmethod
    def is_english(query, ratio=0.5):
        import unicodedata

        new_query = unicodedata.normalize('NFKD', query).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        if len(new_query) >= ratio * len(query):
            return True
        return False

    @staticmethod
    def bi_gram(string):
        bi_g = set()
        string += "$"
        for i in range(len(string)):
            bi_g.add(string[i - 1] + string[i])
        return bi_g

    @staticmethod
    def jac_card(string_1, string_2):
        bi_gram_1, bi_gram_2 = EditQuery.bi_gram(string_1), EditQuery.bi_gram(string_2)
        intersection = set.intersection(bi_gram_1, bi_gram_2)
        union = set.union(bi_gram_1, bi_gram_2)
        return len(intersection) / len(union)

    @staticmethod
    def edit_distance(string_1, string_2):
        matrix = []
        for j in range(len(string_2) + 1):
            a = []
            for i in range(len(string_1) + 1):
                a.append(0)
            matrix.append(a)
        for i in range(len(string_1) + 1):
            matrix[0][i] = i
        for j in range(len(string_2) + 1):
            matrix[j][0] = j
        for i in range(1, len(string_1) + 1):
            for j in range(1, len(string_2) + 1):
                matrix[j][i] = min(
                    matrix[j - 1][i] + 1,
                    matrix[j][i - 1] + 1,
                    matrix[j - 1][i - 1] + (not string_1[i - 1] == string_2[j - 1])
                )
        return matrix[len(string_2)][len(string_1)]

    def edit_query(self):
        query = self.query_token_list
        edited_query = []
        for token in query:
            token_bi_grams = self.bi_gram(token)
            all_related_terms = set()
            for bi in token_bi_grams:
                related_term_for_bi = set(self.indexer.get_bigram_posting(bi))
                all_related_terms = set.union(all_related_terms, related_term_for_bi)
            jac_dic = {}
            for term in all_related_terms:
                jac_dic.update({term: self.jac_card(term, token)})
            jac_dic_sorted = sorted(jac_dic.items(), key=lambda item: item[1])
            top_ten = []
            min_edit_dis = 10000
            selected_term = token
            for i in range(len(jac_dic_sorted) - 1, max(0, len(jac_dic_sorted) - 10), -1):
                jac_term = jac_dic_sorted[i][0]
                distance = self.edit_distance(jac_term, token)
                if distance < min_edit_dis:
                    selected_term = jac_term
                    min_edit_dis = distance
            edited_query.append(selected_term)

        return ' '.join(edited_query)
