import string

def exact_match_score(predictions: list[str], references: list[str]) -> float:
    """
    Calculate the exact match score between predictions and references.
    
    Args:
        predictions: List of predicted strings
        references: List of reference (ground truth) strings
    
    Returns:
        Exact match score as a float between 0 and 1
    """
    if len(predictions) == 0:
        return 0.0
    translator = str.maketrans('', '', string.punctuation)
    # Your code here
    true = 0
    for i in range(len(predictions)):
        temp_pred = predictions[i].lower()
        stripped_pred = temp_pred.translate(translator)
        temp_ref = references[i].lower()
        stripped_ref = temp_ref.translate(translator)
        if " ".join(stripped_pred.split()) == " ".join(stripped_ref.split()):
            true += 1
    accuracy = true/len(predictions)
    return accuracy