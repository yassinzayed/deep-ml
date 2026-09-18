import numpy as np

def calculate_correlation_matrix(X, Y=None):
	if Y is None:
        # Correlation matrix of X with itself
    	return np.corrcoef(X, rowvar=False)
    else:
        # Full combined correlation matrix
        full_matrix = np.corrcoef(X, Y, rowvar=False)
        
        # Extract only the cross-correlation between X and Y
        num_features_x = X.shape[1]
        return full_matrix[:num_features_x, num_features_x:]
