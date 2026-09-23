def random_forest_feature_importance(trees: list, n_features: int) -> list:
    """
    Calculate feature importance from a random forest using Mean Decrease in Impurity.
    
    Args:
        trees: List of trees, where each tree is a list of node splits.
               Each split is a dict with:
               - 'feature_index': int, the feature used for splitting
               - 'impurity_decrease': float, the weighted impurity decrease
        n_features: Total number of features in the dataset
    
    Returns:
        List of feature importances normalized to sum to 1.0
    """
    n = [i for i in range(n_features)]
    n_final = [0 for i in range(n_features)]
    for i in trees:
        for j in i:
            if j['feature_index'] in n:
                n_final[int(j['feature_index'])] += j['impurity_decrease']
    return n_final