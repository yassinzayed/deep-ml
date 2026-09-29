import numpy as np

def double_quantize_scales(scales: np.ndarray):
    """
    Symmetric INT8 absmax quant on a vector of scales.
    scale2 = max|scales|/127 (or 1.0); q = clamp(round(scales/scale2),-127,127)
    Returns q_scales, scale2, scales_hat
    """
    # Your code here
    if all(scales == np.zeros_like(scales)):
        return np.array([0 for i in scales]), 1.0, np.zeros_like(scales)
    scale2 = max(scales)/127
    q = np.clip(np.round(scales/scale2), -127, 127)
    scales_hat = q*scale2
    q = np.array([int(i) for i in q])
    return q, scale2, scales_hat