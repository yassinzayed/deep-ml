import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

	:param dense_matrix: 2D list representing a dense matrix
	:return: A tuple containing (values array, column indices array, row pointer array)
	"""
	values_array = []
	column_indices = []
	row_pointer = [0]
	for i in range(len(dense_matrix)):
		for j in range(len(dense_matrix[i])):
			if dense_matrix[i][j] != 0:
				values_array.append(dense_matrix[i][j])
				column_indices.append(j)
		row_pointer.append(len(values_array))
	return (values_array, column_indices, row_pointer)
