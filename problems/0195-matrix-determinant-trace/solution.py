def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
	"""
	Compute the determinant and trace of a square matrix.
	
	Args:
		matrix: A square matrix (n x n) represented as list of lists
	
	Returns:
		Tuple of (determinant, trace)
	"""
	trace = 0
	for i in range(len(matrix)):
		for j in range(len(matrix)):
			if i == j:
				trace += matrix[i][j]

	def det(matrix, sum=0):
		for k in range(len(matrix)):
			temp_matrix = [l.copy() for l in matrix if (matrix[k]!=l)]
			for i in temp_matrix:
				for j in range(len(i)):
					if j == 0:
						i.pop(j)
			if len(temp_matrix) == 2:
				sum += (-1)**(k)*(matrix[k][0])*(temp_matrix[0][0]*temp_matrix[1][1] - temp_matrix[0][1]*temp_matrix[1][0])
			else:
				sum += (-1)**(k)*(matrix[k][0])*det(temp_matrix)
		return sum


	if len(matrix) > 2:
		determinant = det(matrix)
	else:
		determinant = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
	
	return (determinant, trace)