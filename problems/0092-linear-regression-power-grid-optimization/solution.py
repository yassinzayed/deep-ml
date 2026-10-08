import math

PI = 3.14159

def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
	# 2) Perform linear regression on the detrended data.
	# 3) Predict day 15's base consumption.
	# 4) Add the day 15 fluctuation back.
	# 5) Round, then add a 5% safety margin (rounded up).
	# 6) Return the final integer.
	n = len(consumption_data)
	y = [i+1 for i in range(n)]
	consumption_changed = [i-(10*math.sin(2*PI*r/10)) for i, r in zip(consumption_data, y)]
	denom = n*sum([(i**2) for i in y])-(sum(y)**2)
	nom = n*sum([x * r for x, r in zip(consumption_changed, y)])-(sum(consumption_changed)*sum(y))
	m = nom/denom
	b = (sum(consumption_changed)-m*sum(y))/n
	day = 15
	base = day*m+b
	pred_15 = base+(10*math.sin(2*PI*day/10))
	final = math.ceil(1.05*round(pred_15))
	return final