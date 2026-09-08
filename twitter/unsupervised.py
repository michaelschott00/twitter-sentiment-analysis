"""This script fits a GMM to TF-IDF vectors. It can plot the clusters in 2D using PCA and t-SNE as well as find the clusters with the most negative-, neutral- or positive sentiments."""

from collections import Counter
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from tqdm import tqdm

sns.set_theme()


def load_data():
    df = pd.read_csv(
        "data/splits/tweets_train.csv",
        dtype={
            "author_id": "category",
            "type": "category",
            "possibly_sensitive": "category",
            "sentiment": "category",
        },
    )
    df.words = df.words.apply(
        lambda x: " ".join(
            [
                token
                for token in eval(x)
                if token != "amp" and token != "gt" and token != "lt"
            ]
        )
    ).to_numpy()
    return df


def get_tfidf(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = np.asarray(vectorizer.fit_transform(df.words).todense())
    vocab = np.array(vectorizer.get_feature_names_out())
    return vectors, vocab


def select_k_kmeans(
    df: pd.DataFrame,
    min_k: int = 1,
    max_k: int = 100,
    steps: int = 10,
    feautres: Literal["tfidf"] = "tfidf",
):
    """Mostly taken from https://nkoenig06.github.io/gd-tm-cluster.html"""
    if feautres == "tfidf":
        vectors, vocab = get_tfidf(df)
    else:
        raise ValueError(f"Features {feautres} not supported.")
    if min_k == 1:
        K_list = [1]
        K_list += [i for i in range(min_k + steps - 1, max_k, steps)]
    else:
        K_list = [i for i in range(min_k, max_k, steps)]
    wcss = []
    min_wcss = np.inf
    best_model = None
    for K in tqdm(K_list):
        kmeans = KMeans(
            n_clusters=K, init="k-means++", max_iter=300, n_init=10, random_state=0
        )
        kmeans.fit(vectors)
        if kmeans.inertia_ < min_wcss:
            best_model = kmeans
            min_wcss = kmeans.inertia_
        wcss.append(kmeans.inertia_)
    return best_model, (K_list, wcss)


def select_k_gmm(
    df: pd.DataFrame,
    min_k: int,
    max_k: int,
    steps: int,
    covariance: str = "spherical",
    feautres: Literal["tfidf"] = "tfidf",
):
    if feautres == "tfidf":
        vectors, vocab = get_tfidf(df)
    else:
        raise ValueError(f"Features {feautres} not supported.")
    if min_k == 1:
        K_list = [1]
        K_list += [i for i in range(min_k + steps - 1, max_k, steps)]
    else:
        K_list = [i for i in range(min_k, max_k, steps)]
    BIC = []
    min_bic = np.inf
    best_model = None
    for K in tqdm(K_list):
        gmm = mixture.GaussianMixture(
            n_components=K,
            init_params="k-means++",
            covariance_type=covariance,
            max_iter=300,
            n_init=10,
            random_state=0,
        )
        gmm.fit(vectors)
        bic = gmm.bic(vectors)
        if bic < min_bic:
            best_model = gmm
            min_bic = bic
        BIC.append(bic)

    return best_model, (K_list, BIC)


def add_kmeans_clusters(
    df: pd.DataFrame, n_clusters: int, feautres: Literal["tfidf"] = "tfidf"
):
    if feautres == "tfidf":
        vectors, _ = get_tfidf(df)
    else:
        raise ValueError("Only tfidf is supported for now")
    kmeans = KMeans(
        n_clusters=n_clusters, init="k-means++", max_iter=300, n_init=10, random_state=0
    )
    kmeans.fit(vectors)
    df["cluster"] = kmeans.labels_
    return df


def add_gmm_components(
    df: pd.DataFrame,
    n_components: int,
    covariance: Literal["spherical", "diag"],
    feautres: Literal["tfidf"] = "tfidf",
):
    if feautres == "tfidf":
        vectors, _ = get_tfidf(df)
    else:
        raise ValueError("Only tfidf is supported for now")
    gmm = mixture.GaussianMixture(
        n_components=n_components,
        init_params="k-means++",
        covariance_type=covariance,
        max_iter=300,
        n_init=10,
        random_state=0,
    )
    gmm.fit(vectors)
    df["component"] = gmm.predict(vectors)
    return df


def add_principal_components(
    df: pd.DataFrame,
    kernel=None,
    feautres: Literal["tfidf"] = "tfidf",
    whiten: bool = False,
    center: bool = False,
):
    assert not (kernel is not None and whiten)
    if feautres == "tfidf":
        vectors, _ = get_tfidf(df)
    else:
        raise ValueError("Only tfidf is supported for now")
    if center:
        vectors -= vectors.mean(axis=0)
    if kernel is None:
        pca = PCA(n_components=2, random_state=0, whiten=whiten)
    else:
        pca = KernelPCA(n_components=2, kernel=kernel, random_state=0)
    pca_data = pca.fit_transform(vectors)
    df["pca_1"] = pca_data[:, 0]
    df["pca_2"] = pca_data[:, 1]
    return df


def add_tsne_components(
    df: pd.DataFrame, perplexity: float = 30.0, feautres: Literal["tfidf"] = "tfidf"
):
    if feautres == "tfidf":
        vectors, _ = get_tfidf(df)
    else:
        raise ValueError("Only tfidf is supported for now")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    tsne_data = tsne.fit_transform(vectors)
    df["tsne_1"] = tsne_data[:, 0]
    df["tsne_2"] = tsne_data[:, 1]
    return df


def plot_lower_dimensional(
    df: pd.DataFrame, x: str, y: str, hue_column: str, add_style: bool = False
):
    if add_style:
        sns.scatterplot(
            data=df, x=x, y=y, hue=hue_column, style="sentiment", palette="tab10"
        )
    else:
        sns.scatterplot(data=df, x=x, y=y, hue=hue_column, palette="tab10")
    plt.tight_layout()
    plt.show()


def most_common_words_in_cluster(df: pd.DataFrame, column: str, cluster_id: int):
    counter = Counter()
    for words in df[df[column] == cluster_id].words:
        counter.update(words.split(" "))
    return counter.most_common(3)


def most_polarized(df: pd.DataFrame, column: str, n: int = 3):
    cluster_sentiment = (
        df.groupby(column)["score_compound"]
        .aggregate(np.median)
        .reset_index()
        .sort_values(by="score_compound", ascending=True)
    )
    most_negative = cluster_sentiment[:n]
    most_positive = cluster_sentiment[-n:]
    random_neutral = cluster_sentiment[
        (cluster_sentiment["score_compound"] < 0.05)
        & (cluster_sentiment["score_compound"] > -0.05)
    ].sample(n)
    return pd.concat([most_negative, most_positive, random_neutral])


def select_clusters(df: pd.DataFrame, column: str, cluster_ids: list[int]):
    mask = df[column].isin(cluster_ids)
    return df[mask]


def largest_proportion_of_sentiment(df: pd.DataFrame, sentiment: int):
    sentiment_df = df[df["sentiment"] == sentiment]
    sentiment_count_per_component = sentiment_df["component"].value_counts()
    count_per_component = df["component"].value_counts()
    return (sentiment_count_per_component / count_per_component).idxmax()


# 12, 61, 13
