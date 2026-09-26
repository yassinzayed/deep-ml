import numpy as np

def calculate_contrast(img) -> int:
	"""
	Calculate the contrast of a grayscale image.
	Args:
		img (numpy.ndarray): 2D array representing a grayscale image with pixel values between 0 and 255.
	"""
	# array.reshape(-1) and array.flatten() and array.ravel() flatten the given array
	contrast = max(img.reshape(-1))-min(img.reshape(-1))
	return contrast