def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	trace = matrix[0][0] + matrix[1][1]
	determinant = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
	under_root = trace**2 - 4*determinant
	nom_1 = trace + under_root**(1/2)
	nom_2 = trace - under_root**(1/2)
	denom = 2
	eigenvalues = [nom_1/denom , nom_2/denom]
	return eigenvalues