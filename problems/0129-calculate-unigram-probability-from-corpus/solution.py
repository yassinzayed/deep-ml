def unigram_probability(corpus: str, word: str) -> float:
    corpus_list = corpus.split()
    num = corpus_list.count(word)
    return num/len(corpus_list)