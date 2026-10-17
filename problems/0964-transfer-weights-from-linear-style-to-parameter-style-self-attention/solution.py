import numpy as np

def transfer_and_attend(X, W_q_lin, W_k_lin, W_v_lin):
    """
    Transfer linear-style weights (shape (d_out, d_in)) into parameter-style
    weights (shape (d_in, d_out)) and compute single-head causal self-attention.

    Args:
        X: input array of shape (seq_len, d_in)
        W_q_lin, W_k_lin, W_v_lin: linear-style weight matrices of shape (d_out, d_in)

    Returns:
        Nested list of shape (seq_len, d_out): the self-attention output.
    """
    # Your code here
    X = np.array(X)
    Q = np.dot(X, np.array(W_q_lin).T)
    K = np.dot(X, np.array(W_k_lin).T)
    V = np.dot(X, np.array(W_v_lin).T)
    S = (Q @ K.T)/np.sqrt(np.array(W_q_lin).shape[0])
    # tril is pretty cool
    seq_len = X.shape[0]
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    S = S - (mask * 1e9)
    soft_s = np.exp(S-np.max(S, axis=-1, keepdims=True))/np.sum(np.exp(S-np.max(S, axis=-1, keepdims=True)), axis=-1, keepdims=True)
    output = soft_s @ V
    return output
