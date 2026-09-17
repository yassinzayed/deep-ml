import numpy as np
def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
	# Return the element-wise sum of vectors 'a' and 'b'.
	# If vectors have different lengths, return -1.
	if len(a) == len(b):
		sum_list = [a[i] + b[i] for i in range(len(a))]
		return sum_list
	else:
		return -1