def phi_corr(x: list[int], y: list[int]) -> float:
	"""
	Calculate the Phi coefficient between two binary variables.

	Args:
	x (list[int]): A list of binary values (0 or 1).
	y (list[int]): A list of binary values (0 or 1).

	Returns:
	float: The Phi coefficient rounded to 4 decimal places.
	"""
	n = len(x)
	x_00 = sum(x[i] == 0 and y[i] == 0 for i in range(n))
	x_01 = sum(x[i] == 0 and y[i] == 1 for i in range(n))
	x_10 = sum(x[i] == 1 and y[i] == 0 for i in range(n))
	x_11 = sum(x[i] == 1 and y[i] == 1 for i in range(n))
	nom = (x_00*x_11)-(x_01*x_10)
	denom = (x_00+x_01)*(x_10+x_11)*(x_00+x_10)*(x_01+x_11)
	if denom == 0:
		return 0
	val = nom/(denom)**(1/2)
	return round(val,4)