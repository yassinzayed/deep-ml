import numpy as np
def macro_average_accuracy(subtask_results: dict) -> float:
    """
    Compute the unweighted macro-average accuracy across benchmark subtasks.
    Each value in subtask_results is a list of (prediction, label) tuples.
    Return the macro-average accuracy rounded to 4 decimals.
    """
    if subtask_results == {}:
        return 0.0
    subtask_accuracies = []
    for i in subtask_results:
        accuracy = 0
        for j in subtask_results[i]:
            if len(set(j)) <= 1:
                accuracy += 1
        if len(subtask_results[i]) != 0:
            subtask_accuracies.append(accuracy/len(subtask_results[i]))
        else:
            subtask_accuracies.append(0)
    subtask_final = np.mean(subtask_accuracies)
    return subtask_final