import networkx as nx
import json


class PageRank:
    def page_rank_callculate(self):
        papers = json.load(open('crawler_data/papers3360.json', 'r'))
        paper_ids = {}
        id_title = {}
        graph = nx.DiGraph()
        for i in range(len(papers)):
            graph.add_node(i)
            paper_ids[papers[i]['id']] = i
            id_title[i] = (papers[i]['title'], papers[i]['id'])

        for paper in papers:
            refs = paper['references']
            for ref in refs:
                ref_id = paper_ids.get(ref.split('/')[4].split('\n')[0])
                if ref_id is None:
                    continue
                graph.add_edge(paper_ids[paper['id']], ref_id)

        alpha = float(input('specify alpha: '))
        page_rank = nx.pagerank(graph, alpha)
        doc_ranks = []
        for id, rank in page_rank.items():
            doc_ranks.append((id_title[id], rank))
        doc_ranks.sort(key=lambda tup: tup[1], reverse=True)
        return doc_ranks


if __name__ == '__main__':
    page_rank = PageRank()
    print(page_rank.page_rank_callculate())
