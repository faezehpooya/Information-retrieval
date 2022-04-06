from Phase3.preprocess import read_and_process, load_processed_data_from_file
from Phase3.Tf_idf_vectorizer import Tf_idf_vectorizer
from Phase3.Word2vec import Word2Vec
from Phase3.K_Means import Kmeans
from Phase3.GMM import GMM
from Phase3.Hierarchical_Clustering import HierarchicalClustering
from Phase3.Cluster import Cluster


if __name__ == '__main__':
    data = read_and_process()
    vectorizers = [Tf_idf_vectorizer, Word2Vec]
    clustering_models = [Kmeans, GMM, HierarchicalClustering]

    for model in clustering_models:
        for vectorizer in vectorizers:
            cluster = Cluster(data=data, vectorizer=vectorizer, model=model)
            print("->", cluster.model.method_name, "clustering algorithm and", cluster.vectorizer.name, "vectorizer")
            print("   Purity score:", cluster.get_purity(), ", Rand Index score:", cluster.get_rand_index(),
                  ", Adjusted Mutual Info score:", cluster.AMI_score(),
                  ", Adjusted Rand Index score:", cluster.adjusted_rand_index_score())
            cluster.model.plot()
            cluster.plot_tsne()
            cluster.report()

        print("________________")
