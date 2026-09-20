
def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	def dot_prod(a, b):
		sums = 0
		for i in range(len(a)):
			sums += a[i]*b[i]
		return sums
	nom = dot_prod(v, L)
	denom = dot_prod(L, L)
	proj = [x*(nom/denom) for x in L]
	return proj
