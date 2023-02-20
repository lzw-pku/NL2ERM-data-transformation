from collections import defaultdict
from evaluate.utils import match, ngram_similarity

def compute_similarity(name_set, name):
    return max([ngram_similarity(n, name) for n in name_set])

def compute_entity_similarity(e1, e2):
    score1 = compute_similarity(e1.name_set, e2.name)
    similarity = defaultdict(dict)
    for a1 in e1.attribute:
        for a2 in e2.attribute:
            similarity[a1][a2] = compute_similarity(a1, a2)
    score2, match_result = match(similarity)
    return (score1 + score2) / (1 + max(len(e1.attribute), len(e2.attribute)))

def metric(erd_ref, erd_pred):
    similarity = defaultdict(dict)
    for e1 in erd_pred.entities:
        for e2 in erd_ref.entities:
            similarity[e1][e2] = compute_entity_similarity(e1, e2)
    score, match_result = match(similarity, threshold=0.3)
    print(match_result)
