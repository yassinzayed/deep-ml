import numpy as np

def bernoulli_mask_schedule(seq_len: int, t: float, schedule: str, seed: int) -> list:
	"""
	Generate a Bernoulli mask whose probability is set by a schedule.

	Args:
		seq_len: length of the token sequence
		t: schedule parameter in [0, 1]
		schedule: 'linear' or 'cosine'
		seed: random seed

	Returns:
		A list of 0/1 integers of length seq_len (1 = masked).
	"""
	np.random.seed(seed)
	random_arr = np.random.rand(seq_len)
	if schedule == "linear":
		arr = np.where(random_arr<t, 1, 0)
	elif schedule == "cosine":
		arr = np.where(random_arr<(1-np.cos((np.pi*t)/2)), 1, 0)
	return arr 