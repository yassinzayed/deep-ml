def compressed_col_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

	:param dense_matrix: List of lists representing the dense matrix
	:return: Tuple of (values, row indices, column pointer)
	"""
	dense_matrix_t = [list(row) for row in zip(*dense_matrix)]
	values = []
	row_indices = []
	column_pointer = [0]
	for i in range(len(dense_matrix_t)):
		for j in range(len(dense_matrix_t[i])):
			if dense_matrix_t[i][j] != 0:
				values.append(dense_matrix_t[i][j])
				row_indices.append(j)
		column_pointer.append(len(values))
	return (values, row_indices, column_pointer)
