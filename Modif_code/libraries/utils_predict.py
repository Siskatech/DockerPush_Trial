import numpy as np
import re
from .utils_tfIdf import score_tfidf
from collections import defaultdict, Counter

class TextPredictor:
    def __init__(self, corpus, stop_words=None):
        self.corpus = corpus
        self.stop_words = stop_words if stop_words else []
        self.tfidf_scores, self.terms = score_tfidf(corpus, self.stop_words)
        self.term_to_index = {term: idx for idx, term in enumerate(self.terms)}
        self.co_occurrence_matrix = self.build_co_occurrence_matrix(corpus)

    def build_co_occurrence_matrix(self, corpus, window_size=2):
        co_occurrence = defaultdict(Counter)
        for text in corpus:
            words = re.findall(r'\b\w+\b', text.lower())
            for i, word in enumerate(words):
                start = max(0, i - window_size)
                end = min(len(words), i + window_size + 1)
                for j in range(start, end):
                    if i != j:
                        co_occurrence[word][words[j]] += 1
        return co_occurrence

    
    def predict_next_words(self, input_text, top_n=5, max_words=6):
        input_words = re.findall(r'\b\w+\b', input_text.lower())
        predictions = []

        for i in range(len(input_words)):
            words_to_process = input_words[i:]
            query_word = ' '.join(words_to_process)
            
            if query_word in self.co_occurrence_matrix:
                most_common_words = self.co_occurrence_matrix[query_word].most_common(top_n)
                for next_word, _ in most_common_words:
                    next_word_index = self.term_to_index.get(next_word)
                    
                    if next_word_index is not None and next_word not in input_words:
                        tfidf_score = self.tfidf_scores[next_word_index]
                        predictions.append((next_word, tfidf_score))

        predictions.sort(key=lambda x: x[1], reverse=True)

        final_predictions = []
        for i in range(min(top_n, len(predictions))):
            current_word, current_score = predictions[i]
            combined_prediction = current_word
            
            words_added = 1
            for j in range(i+1, len(predictions)):
                if words_added >= max_words - 1:
                    break
                next_word, _ = predictions[j]
                combined_prediction += ' ' + next_word
                words_added += 1

            final_predictions.append((combined_prediction, current_score))
        
        return final_predictions[:top_n] if final_predictions else [("No prediction available", 0.0)]
