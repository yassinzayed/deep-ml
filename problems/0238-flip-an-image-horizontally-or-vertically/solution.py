import numpy as np

def flip_image(image, direction):
    """
    Flip an image horizontally or vertically.
    
    Args:
        image: 2D or 3D list/array representing a grayscale or RGB image
        direction: string, either 'horizontal' or 'vertical'
    
    Returns:
        Flipped image as a nested list, or -1 if input is invalid
    """
    if direction == "horizontal":
        return np.flip(image, axis=1)
    elif direction == "vertical":
        return np.flip(image, axis=0)
    else:
        return -1