# helper functions for member meeting workshop - Spring 2021

from curses import def_shell_mode
import numpy as np
import matplotlib as mpl
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from collections import defaultdict


mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = True



"""
From last sessions we have:

They can also get this in one step by importing and using 
`TfidfVectorizer` from sklearn.feature_extraction.text
"""

from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer 
stemmer = PorterStemmer()

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


def get_documents_embedding(documents, stemmed=False, smooth_idf=False,
                            norm=None, lowercase=True, stop_words='english',
                            as_df=False, index=None):
    
    # We select and define what counter we'll be using
    params = dict(lowercase=lowercase, stop_words=stop_words,
                  norm=norm, smooth_idf=smooth_idf)

    counter = TfidfVectorizer(**params)
    tf_idf = counter.fit_transform(documents)


    tf_idf = tf_idf.toarray()

    if as_df:
        feature_names = counter.get_feature_names()
        documents_embeddings = pd.DataFrame(tf_idf, columns=feature_names, index=index)
        return documents_embeddings

    return tf_idf


from sklearn.decomposition import PCA

def reduce_dimensionality(docs_embeddings, n_components=30):
    """
    Parameters
    ----------
    docs_embeddings: shape=(n_docs, n_words),
        2-D array with documents representations, such as TF-IDF.
    n_components: output dimensionality.
                  How many dimension do the new vectors have?
    Returns
    -------
    reduced_embeddings: shape=(n_docs, n_components)
                        2-D array with reduced documents representations.
    """

    initial_shape = docs_embeddings.shape
    n_documents, n_features = initial_shape

    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(docs_embeddings)
    variance_ratio = pca.explained_variance_ratio_.sum().round(4)*100
    
    # final_shape = reduced_embeddings.shape

    # print(f"Each document is represented by {n_features} dimensions.")
    # print("In such high dimension is hard to find dense sub-spaces.")
    # print("We apply a linear transformation to \"collapse\" our data into " + \
    #       f"{n_components} dimensions.")
    # print(f"You've used PCA to go from {initial_shape} to {final_shape}.")
    print("This dimensionality reduction maitains {:.2f}% of the ".format(variance_ratio) +\
          "original variance (information).\n")

    return reduced_embeddings


from sklearn import manifold

def get_visualization_coordinates(docs_reduced_embeddings):
    """
    Parameters
    ----------
    docs_embeddings: shape=(n_docs, n_words),
                    2-D array with documents representations, such as TF-IDF.
    Returns
    -------
    viz_coordinates : shape=(n_docs, 2)
                    2-D array with coordinates (x, y) to visualize documents in
                    a 2-D scatter plot.
    """


    tsne = manifold.TSNE()
    viz_coordinates = tsne.fit_transform(docs_reduced_embeddings)
    return viz_coordinates

from sklearn.cluster import KMeans
from scipy.cluster.vq import vq

def get_clusters(docs_embeddings, n_clusters=10):
    """
    Parameters
    ----------
    docs_embeddings: shape=(n_docs, n_dimensions),
                     2-D array with documents representations, such as TF-IDF.
    n_clusters: how many clusters we'll search for.
    Returns
    -------
    X_clustered: shape=(n_docs,)
                        1-D array with a cluster's assignation per document.
    """

    clustering_model = KMeans(n_clusters=n_clusters) 
    predictions = clustering_model.fit_predict(docs_embeddings)
    centers = clustering_model.cluster_centers_
    closest, distances = vq(centers, docs_embeddings)

    return closest, predictions


import plotly.graph_objs as go


def make_paragraph(text, words_per_line=10):
    """
    Helper function takes a text and after every `words_per_line`
    insters "<br>" so that our visualization shows multi-line
    text and not all text in one line.
    
    Parameters
    ----------
    words_per_line: int that define the n. of words per sentence
    
    Returns
    -------
    paragraph: string with a <br> every `words_per_line` "words"

    """
    _words = text.split()
    word_packages = []
    for i,_ in list(enumerate(_words))[::words_per_line]:
        word_packages.append(_words[i:i+words_per_line])
    lines = [" ".join(package) for package in word_packages]
    paragraph = "<br>".join(lines)
    return paragraph


def plot_tsne_viz(viz, texts=None, centers=None,
                  clusters=None, labels=None, coloring=None,
                  title='k-means clustering',
                  line_color='darkred', marker_size=5, colorscale='sunset',
                  opacity=0.7, text_hook=make_paragraph, labels_text=None):
    
    """
    Plotting function as input a 2-d array (viz) to be drawn in a 2-D graph.
    
    Parameters
    ----------
    """
    
    color = clusters
    if coloring == 'labels':
        color = labels
    # if centers:
    #     marker_size = [10 if c==1 else 5 for c in centers]
    #     marker_symbols = ["star" if c==1 else "circle" for c in centers]
        
    line_width = 0
    if not labels is None:
        class_a = min(labels)
        line_width = [1.5 if val != class_a else 0 for val in labels] if coloring is None else 0

    showscale = False
    if (not clusters is None and coloring != 'labels') or (coloring == 'labels'):
        showscale = True
    
    hovertext = texts
    if not text_hook is None and not texts is None:
        hovertext = [text_hook(x) for x in texts]

    colorbar = None
    if not color is None:
        color_names = list(set(color))
        if not labels_text is None:
            color_names = labels_text
        color_vals = list(range(len(color_names)))
        colorbar = {'title': coloring,
                'tickvals': color_vals,
                'ticktext': color_names,
        }

    plot_data = go.Scatter(
        x = viz[:, 0],
        y = viz[:, 1],
        mode = 'markers',
        showlegend = False,
        hovertext = hovertext,
        marker = dict(
            colorbar = colorbar,
            showscale = showscale,
            colorscale = colorscale,
            # symbol = marker_symbols,
            size = marker_size,
            color = color,
            line = {'width': line_width,
                    'color': line_color},
            opacity = opacity
        )
    )
    fig = go.Figure(data=plot_data, layout_title_text=title)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()

def set_replicable_results(replicable, seed=222):
    """
    Whenever we run a model or piece of code that uses any 
    kind of randomness, we should expect to get different 
    results every time that we ran that piece of code.
    In order to get replicable results we need to specify
    a `seed` (a seed for the random number generation
    process.)
    """
    if replicable:
        np.random.seed(seed)
    else:
        np.random.seed(None)

def run_clustering(data, pca_components, k_clusters, embeddings=None):

    if embeddings is None:
        # use tf-idf to get document embeddings
        documents_embeddings = get_documents_embedding(data)
    else:
        # receive document embeddings
        documents_embeddings = embeddings

    # reduce dimensionality to use a more dense space for clustering
    if embeddings.shape[1] > pca_components:
        reduced_embeddings = reduce_dimensionality(documents_embeddings, n_components=pca_components)
    else:
        reduced_embeddings = embeddings

    # run K-Means to find cluster assignations
    centers, predicted_clusters = get_clusters(reduced_embeddings, n_clusters=k_clusters)

    # reduces data to 2-D coordinates for visualization purposes
    viz_coord = get_visualization_coordinates(reduced_embeddings)

    return (viz_coord, centers, predicted_clusters,
            documents_embeddings, reduced_embeddings)

from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_util


def retrieve(sent_model, queries, corpus, corpus_embs, closest_n=8):
    """
    For each query, print closest matching documents in the corpus.

    Args:
        sent_model: SentenceTransformer model
        queries: list of strings
        corpus: list of strings, length N
        corpus_embs: [N, D] tensor
        closest_n: int
    
    Return:
        a dataframe of similar sentences of the input queries, 
        similarity score noted by a score column
        cluster noted by a cluster column
    """
    dfs = []
    for num, query in enumerate(queries):
        query_embedding = sent_model.encode(query, convert_to_tensor=True)
        scores = st_util.pytorch_cos_sim(query_embedding, corpus_embs)[0]

        results = zip(range(len(scores)), scores)
        results = sorted(results, key=lambda x: x[1], reverse=True)

        result_utts = []
        scores = []

        for idx, score in results[0:closest_n]:
            result_utts.append(corpus[idx].strip())
            scores.append(score)
        df = pd.DataFrame([result_utts,scores]).T
        df.columns=["similar_sentences","scores"]
        df["cluster"] = num
        dfs.append(df)
    return pd.concat(dfs)
    

    

def parse_emolex(file_path=None):
    """
    Returns:
        word2sentiments: dict
            key: str
            value: set of strs
        word2emotions: dict
            key: str
            value: set of strs
    """
    if file_path is None:
        file_path = 'NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'

    word2sentiments = defaultdict(set)
    word2emotions = defaultdict(set)
    f = open(file_path, 'r')
    for i, line in enumerate(f.readlines()):
        if i > 1:
            # line: 'abandoned\tanger\t1\n',
            word, emotion, flag = line.strip('\n').split('\t')
            if int(flag) == 1:
                if emotion == 'positive' or emotion == 'negative':
                    word2sentiments[word].add(emotion)
                else:
                    word2emotions[word].add(emotion)
    f.close()

    return word2sentiments, word2emotions

def vader_based_dt_analysis(conversations):
    """
    Args:
      conversations: list of dicts

    Return
        dict: key is date, value is dict D
            D: key is sentiment/emotion, value is count
    """
    dt2sentiment = defaultdict(float)
    dt2counts = defaultdict(int)
    for conversation in conversations:
        for i, utterance in enumerate(conversation):
            sentiment = utterance['compound']
            dt2sentiment[i] += sentiment
            dt2counts[i] += 1

    # normalize
    dt2sentiment = {dt: sent / dt2counts[dt] for dt, sent in dt2sentiment.items()}
    return dt2sentiment

from sklearn.neighbors import KNeighborsRegressor

def plot_conv_valence(dt2sentiment, hovertext, title_prepend='', show_trend_line=False,
         trend_steps=50, trend_knn=3):
    """
    Args:
        dt2sentiment: output of vader_based_dt_analysis()
        hovertext: list of text matching order in dt2sentiment.
        title_prepend: string to be prepend to the title.
        show_trend_line: whether to show a trend line.
        trend_steps: how many x points should be used for approximation.
        trend_knn: howm any neighbors valences to use for trend estimation.
    """
    # prep
    dt_sentiment = sorted(dt2sentiment.items(), key=lambda x: x[0])
    dts, sentiments = zip(*dt_sentiment)
    dts, sentiments = np.array(list(dts)), np.array(list(sentiments))

    # Plot
    plot_data = go.Scatter(
        x = dts,
        y = sentiments,
        mode = 'markers',
        showlegend = False,
        hovertext = hovertext,
        marker = dict(
            opacity = 1,
            color= 1 - np.array(sentiments),
            colorscale='bluered',
            size=10
        ),
    )

    fig = go.Figure(data=plot_data)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(yaxis_range=[-1,1])

    if show_trend_line:
        x_range = np.linspace(min(dts), max(dts), trend_steps)
        knn_dist = KNeighborsRegressor(trend_knn, weights='distance')
        knn_dist.fit(dts.reshape(-1, 1), sentiments.reshape(-1, 1))
        y_dist = knn_dist.predict(x_range.reshape(-1, 1)).T[0]
        fig.add_traces(go.Scatter(x=x_range, y=y_dist,
                                name=f'Weighted\n{trend_knn}-Neighbors',
                                line=dict(color="gray", width = 2)))

    fig.update_layout(
        title=f"{title_prepend}Emotional valence through a conversation.",
        xaxis_title="Time through conversation (~ 60-90 mins)",
        yaxis_title="Emotional Valence",
    )
    fig.show()





