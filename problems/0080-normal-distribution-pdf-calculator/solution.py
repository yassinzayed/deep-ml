import math

def normal_pdf(x, mean, std_dev):
	"""
	Calculate the probability density function (PDF) of the normal distribution.
	:param x: The value at which the PDF is evaluated.
	:param mean: The mean (μ) of the distribution.
	:param std_dev: The standard deviation (σ) of the distribution.
	"""
	# Your code here
	denom = math.sqrt(2*math.pi*(std_dev**2))
	exp_power = -((x-mean)**2/(2*(std_dev**2)))
	val = (1/denom)*math.exp(exp_power)
	return round(val,5)