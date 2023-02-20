import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
#from transformers import BertTokenizer, BertModel
import torch
#import bert_score
#score = bert_score.BERTScorer('microsoft/deberta-xlarge-mnli',
#                              lang='en', rescale_with_baseline=True)


def match(mat, threshold=0.5, punish=-1):
    if len(mat) == 0: return 0, {}
    name1 = list(mat.keys())
    name2 = list(mat[name1[0]].keys())
    t = [[-mat[n1][n2] if mat[n1][n2] > threshold else -punish
          for n2 in name2] for n1 in name1]
    t = np.asarray(t)
    row_index, column_index = linear_sum_assignment(t)
    score = 0
    match_result = defaultdict()
    for row, column in zip(row_index, column_index):
        n1 = name1[row]
        n2 = name2[column]
        score += t[row][column]
        match_result[n1] = n2
    score -= abs(len(name1) - len(name2)) * punish
    return -score, match_result


def ngram_similarity(ngram1, ngram2):
    #p, r, f = score.score([ngram1], [ngram2])
    ngram1 = ngram1.split()
    ngram2 = ngram2.split()
    num = 0
    for x in ngram1:
        if x in ngram2:
            num += 1
    for x in ngram2:
        if x in ngram1:
            num += 1
    return num / (len(ngram1) + len(ngram2))
    #return max(p[0], r[0], f[0])

#print(match(None))


#t = np.asarray([[1.,2.],  [1.,2.], [4., 5.]])
#print(linear_sum_assignment(t))

#print(match({'x': {'a':1, 'b':2}, 'y':{'a':10, 'b':5}, 'z':{'a': 3, 'b':8}}))

def merge(corpus):
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    import numpy as np

    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    corpus_embeddings = embedder.encode(corpus)

    # Normalize the embeddings to unit length
    corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    # Perform kmean clustering
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5) #, affinity='cosine', linkage='average', distance_threshold=0.4)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(corpus[sentence_id])

    for i, cluster in clustered_sentences.items():
        print("Cluster ", i+1)
        print(cluster)
        print("")
