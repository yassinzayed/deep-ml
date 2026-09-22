import numpy as np
from collections import Counter

def dummy_classifier(y_train, n_test, strategy, constant=None):
    """
    Produce baseline predictions of length n_test using the given strategy.
    Returns a Python list of predicted labels.
    """
    if strategy == "most_frequent":
        element = max(set(y_train), key=y_train.count)
        predicted = [element for i in range(n_test)]
    elif strategy == "constant":
        predicted = [constant for i in range(n_test)]
    elif strategy == "uniform":
        predicted = [0 for i in range(n_test)]
        sorted_classes = sorted(y_train)
        for i in range(len(predicted)):
            predicted[i] = sorted_classes[i % len(set(y_train))]
    elif strategy == "stratified":
        predicted = y_train[:n_test]
    return predicted
