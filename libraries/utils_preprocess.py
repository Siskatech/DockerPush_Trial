import pandas as pd
import numpy as np
import nltk

def get_CorpusData(verbose=False):
    with open("data/idwiki.txt", "r", encoding='UTF-8') as f:
        data = f.read()
    if verbose:
        print("Data type:", type(data))
        print("Number of letters:", len(data))
        print("-------\nFirst 300 letters of the data\n-------")
        display(data[0:300])
        print("-------\nLast 300 letters of the data\n-------")
        display(data[-300:])

    return data

def split_to_sentences(data):
    """
    Split data by linebreak "\n"
    Args:
        data: str
    Returns:
        A list of sentences
    """
    sentences = data.split('\n')
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0]

    return sentences

def tokenize_sentences(sentences):
    """
    Tokenize sentences into tokens (words)
    Args:
        sentences: List of strings
    Returns:
        List of lists of tokens
    """
    tokenized_sentences = []

    for sentence in sentences:
        sentence = sentence.lower()
        tokenized = nltk.word_tokenize(sentence)
        tokenized_sentences.append(tokenized)

    return tokenized_sentences

def get_tokenized_data(data):
    """
    Make a list of tokenized sentences
    Args:
        data: String
    Returns:
        List of lists of tokens
    """
    sentences = split_to_sentences(data)
    tokenized_sentences = tokenize_sentences(sentences)
    return tokenized_sentences

def count_words(tokenized_sentences):
    """
    Count the number of word appearence in the tokenized sentences
    Args:
        tokenized_sentences: List of lists of strings
    Returns:
        dict that maps word (str) to the frequency (int)
    """
    word_counts = {}

    for sentence in tokenized_sentences:
        for token in sentence:
            if token not in word_counts.keys():
                word_counts[token] = 1
            else:
                word_counts[token] += 1

    return word_counts

def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):
    """
    Find the words that appear N times or more
    Args:
        tokenized_sentences: List of lists of sentences
        count_threshold: minimum number of occurrences for a word to be in the closed vocabulary.
    Returns:
        List of words that appear N times or more
    """
    closed_vocab = []
    word_counts = count_words(tokenized_sentences)

    for word, cnt in word_counts.items():
        if cnt >= count_threshold:
            closed_vocab.append(word)

    return closed_vocab

def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    """
    Replace words not in the given vocabulary with '<unk>' token.
    Args:
        tokenized_sentences: List of lists of strings
        vocabulary: List of strings that we will use
        unknown_token: A string representing unknown (out-of-vocabulary) words
    Returns:
        List of lists of strings, with words not in the vocabulary replaced
    """
    vocabulary = set(vocabulary)
    replaced_tokenized_sentences = []

    for sentence in tokenized_sentences:
        replaced_sentence = []
        for token in sentence:
            if token in vocabulary:
                replaced_sentence.append(token)
            else:
                replaced_sentence.append(unknown_token)

        replaced_tokenized_sentences.append(replaced_sentence)

    return replaced_tokenized_sentences

def preprocess_data(train_data, test_data, count_threshold):
    """
    Preprocess data, i.e.,
        - Find tokens that appear at least N times in the training data.
        - Replace tokens that appear less than N times by "<unk>" both for training and test data.
    Args:
        train_data, test_data: List of lists of strings.
        count_threshold: Words whose count is less than this are treated as unknown.
    Returns:
        Tuple of
        - training data with low frequent words replaced by "<unk>"
        - test data with low frequent words replaced by "<unk>"
        - vocabulary of words that appear n times or more in the training data
    """
    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)
    train_data_replaced = replace_oov_words_by_unk(train_data, vocabulary)
    test_data_replaced = replace_oov_words_by_unk(test_data, vocabulary)
    return train_data_replaced, test_data_replaced, vocabulary