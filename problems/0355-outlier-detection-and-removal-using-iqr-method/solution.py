import numpy as np

def detect_outliers_iqr(data: list[float], k: float = 1.5) -> dict:
	"""
	Detect and remove outliers using the IQR method.
	
	Args:
		data: List of numerical values
		k: IQR multiplier for determining outlier bounds (default 1.5)
	
	Returns:
		Dictionary with 'cleaned_data', 'outlier_indices', 'lower_bound', 'upper_bound'
	"""
	q75, q25 = np.percentile(data, [75, 25])
	iqr = q75 - q25
	lower_bound = q25-(k*iqr)
	upper_bound = q75+(k*iqr)
	clean_data = data.copy()
	indices = []
	for i in data:
		if i < lower_bound or i > upper_bound:
			clean_data.remove(i)
			indices.append(data.index(i))
	return {
		'cleaned_data': clean_data,
		'outlier_indices': indices,
		'lower_bound': lower_bound,
		'upper_bound': upper_bound
	}