def calculate_brightness(img):
	# improve code!!
	if img == [] or (sum(len(i) != len(img[0]) for i in img) >= 1):
		return -1
	sums = 0
	n = 0
	for i in img:
		for j in i:
			if j > 255 or j < 0:
				return -1
			sums += j
			n += 1
	r = sums/n
	return r
