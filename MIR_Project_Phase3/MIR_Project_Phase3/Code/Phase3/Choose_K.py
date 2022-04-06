from sklearn.cluster import KMeans
from Phase3.preprocess import load_processed_data_from_file
from Phase3.Tf_idf_vectorizer import Tf_idf_vectorizer
from Phase3.Word2vec import Word2Vec
import matplotlib.pyplot as plt
from kneed import KneeLocator

vectorizers = [Tf_idf_vectorizer, Word2Vec]

kmeans_kwargs = {
    "init": "k-means++",
    "max_iter": 1000,
}

# A list holds the SSE values for each k
sse = [[] for i in range(len(vectorizers))]
max_k = 30
i = 0
data = load_processed_data_from_file()
for vectorizer in vectorizers:
    vectorizer = vectorizer(data['text'])
    data_vectors = vectorizer.get_vectors(data['text'])
    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data_vectors)
        sse[i].append(kmeans.inertia_)

    # plt.style.use("fivethirtyeight")
    plt.plot(range(1, max_k), sse[i])
    plt.xticks(range(1, max_k))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.title(vectorizer.name)
    plt.show()

    """
    When you plot SSE as a function of the number of clusters, notice that SSE continues to decrease as you increase k. 
    As more centroids are added, the distance from each point to its closest centroid will decrease.

    There’s a sweet spot where the SSE curve starts to bend known as the elbow point. 
    The x-value of this point is thought to be a reasonable trade-off between error and number of clusters.
    
    """

    kl = KneeLocator(range(1, max_k), sse[i], curve="convex", direction="decreasing")
    print("the elbow point for", vectorizer.name, "is:", kl.elbow)

    i += 1
