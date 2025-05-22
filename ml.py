from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import umap
from keybert import KeyBERT
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np


# Preprocessing: lowercasing, removing stopwords, lemmatization
def preprocess(texts, nlp, stop_words,):

    processed = []
    for text in texts:
        doc = nlp(text.lower()) 
        tokens = [token.lemma_ for token in doc if token.lemma_ not in stop_words and token.is_alpha]
        processed.append(" ".join(tokens))
    return processed

# Embedding: using SentenceTransformer
def embedding(preprocessed_text: str, model_name: str):
    model = SentenceTransformer(model_name)
    return model.encode(preprocessed_text), model

# Dimensionality Reduction: using UMAP
def reduce_dimensionality(embeddings):
    reducer = umap.UMAP(n_neighbors=3, min_dist=0.3, metric='cosine', random_state=42)
    return reducer.fit_transform(embeddings)

# Clustering: using KMeans
def cluster_embeddings(embedding_2d, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(embedding_2d)

# Labeling: using KeyBERT
def label_clusters(model, n_clusters, clusters, input_ideas):
    kw_model = KeyBERT(model=model)
    cluster_texts = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(clusters):
        cluster_texts[label].append(input_ideas[idx])

    cluster_labels_text = []
    for texts in cluster_texts:
        joined = ". ".join(texts)
        keywords = kw_model.extract_keywords(joined, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=1)
        label = keywords[0][0] if keywords else "Cluster"
        cluster_labels_text.append(label)

    return cluster_labels_text

# Plotting: using Matplotlib
def plot_clusters(plot, embedding_2d, clusters, cluster_labels_text, input_ideas):
    # plot.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'green']
    for i, (x, y) in enumerate(embedding_2d):
        label = clusters[i]
        plot.scatter(x, y, color=colors[label], label=cluster_labels_text[label], s=100, alpha=0.6)
        plot.text(x + 0.01, y + 0.01, input_ideas[i], fontsize=9)

    for label_idx in range(len(cluster_labels_text)):
        points = np.array([embedding_2d[i] for i in range(len(embedding_2d)) if clusters[i] == label_idx])
        if len(points) >= 3:  # Convex hull needs at least 3 points
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                plot.plot(points[simplex, 0], points[simplex, 1], color=colors[label_idx], linewidth=2, alpha=0.4)
            plot.fill(points[hull.vertices,0], points[hull.vertices,1], color=colors[label_idx], alpha=0.1)


    # Create legend with cluster labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, 
                        markerfacecolor=color, markersize=10) 
            for label, color in zip(cluster_labels_text, colors)]
    plot.legend(handles=handles, title="Clusters", loc="best")
    plot.set_title("Semantic Clustering of Ideas with UMAP + KMeans")
    plot.set_xlabel("UMAP Dim 1")
    plot.set_ylabel("UMAP Dim 2")
    plot.grid(True)


def ml_pipeline(plot, nlp, stop_words, input_ideas):
    preprocessed_text = preprocess(input_ideas, nlp, stop_words,)
    
    embedding_model_name = "all-MiniLM-L6-v2"

    embeddings, model = embedding(preprocessed_text, embedding_model_name)
    embedding_2d = reduce_dimensionality(embeddings)

    n_clusters = 3
    clusters = cluster_embeddings(embedding_2d, n_clusters)
    labels = label_clusters(model, n_clusters, clusters, input_ideas)
    plot_clusters(plot, embedding_2d, clusters, labels, input_ideas)