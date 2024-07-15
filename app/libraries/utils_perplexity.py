from libraries import utils_ngram as ung

def calculate_perplexity(sentence, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    """
    Calculate perplexity for a list of sentences
    Args:
        sentence: List of strings
        n_gram_counts: Dictionary of counts of (n+1)-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary_size: number of unique words in the vocabulary
        k: Positive smoothing constant
    Returns:
        Perplexity score
    """
    n = len(list(n_gram_counts.keys())[0])
    sentence = ["<s>"] * n + sentence + ["<e>"]
    sentence = tuple(sentence)
    N = len(sentence)
    product_pi = 1.0

    for t in range(n, N):
        n_gram = sentence[t-n:t]
        word = sentence[t]
        probability = ung.estimate_probability(word, n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=k)
        product_pi *= 1/probability

    perplexity = product_pi**(1/N)

    return perplexity