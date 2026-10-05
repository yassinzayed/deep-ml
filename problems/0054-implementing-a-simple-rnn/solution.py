import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	# Your code here
	input_sequence = np.array(input_sequence)
	hidden_state = np.array(initial_hidden_state)
	Wx, Wh, b = np.array(Wx), np.array(Wh), np.array(b)
	for i in input_sequence:
		hidden_state = np.tanh(np.matmul(Wx, i)+np.matmul(Wh, hidden_state)+b)
	return hidden_state