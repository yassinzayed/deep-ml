def hard_voting_classifier(predictions: list[list[int]]) -> list[int]:
    """
    Implement a hard voting classifier using majority vote.
    
    Args:
        predictions: 2D list where predictions[i][j] is classifier i's prediction for sample j
        
    Returns:
        List of final predictions using majority vote
    """
    transposed = [list(row) for row in zip(*predictions)]
    pred = [max(set(i), key=i.count) for i in transposed]
    return pred