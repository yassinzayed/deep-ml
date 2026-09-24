def calculate_parameters(layers: list[dict]) -> int:
	"""
	Calculate the total number of trainable parameters in a neural network.

	Args:
		layers: List of dictionaries, each describing a layer.

	Returns:
		Total number of trainable parameters (int).
	"""
	params = 0
	for i in layers:
		if i['type'] == 'dense':
			if ('bias' in i) and i['bias'] == False:
				params += i['input_size']*i['output_size']
			else:
				params += i['input_size']*i['output_size']+i['output_size']
		elif i['type'] == 'conv2d':
			if ('bias' in i) and i['bias'] == False:
				params += i['in_channels']*i['out_channels']*i['kernel_size']**2
			else:
				params += i['in_channels']*i['out_channels']*i['kernel_size']**2+i['out_channels']
	return params