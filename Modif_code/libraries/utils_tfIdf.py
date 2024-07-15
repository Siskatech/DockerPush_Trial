import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def score_tfidf(corpus,stop_words=None):
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    terms = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    max_score_index = tfidf_scores.argmax()
    max_score = tfidf_scores[max_score_index]
    tfidf_score = tfidf_scores / max_score 

    sorted_indices = np.argsort(-tfidf_score)
    terms = [terms[i] for i in sorted_indices]
    tfidf_scores = tfidf_score[sorted_indices]

    return tfidf_scores ,terms