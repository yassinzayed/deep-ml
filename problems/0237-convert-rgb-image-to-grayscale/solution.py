import numpy as np

def rgb_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminosity method.
    
    Args:
        image: RGB image as list or numpy array of shape (H, W, 3)
               with values in range [0, 255]
    
    Returns:
        Grayscale image as 2D list with integer values,
        or -1 if input is invalid
    """
    # Write your code here
    if len(image) == 0:
        return -1
    if any(len(row) == 0 for row in image):
        return -1
    if any(
        not isinstance(pixel, (list, tuple)) or len(pixel) != 3
        for row in image
        for pixel in row):
        return -1
    if any(value > 255 or value < 0
    for row in image
    for pixel in row
    for value in pixel):
        return -1
    gray = []
    for i in image:
        current_row = []
        for k in i:
            current_row.append(round(0.299*k[0]+0.587*k[1]+0.114*k[2]))
        gray = gray+[current_row]
    return gray