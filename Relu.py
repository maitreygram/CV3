import torch

class ReLu:
	def __init__(self):
		# batch = 1
		self.output = None
		self.gradInput = None

	def forwardpass(self, input):
		self.output = myReLU(input)
		return self.output

	def backwardpass(self, input, gradOutput, learning_rate):
		self.gradInput = gradOutput * (input > 0).double()
		return self.gradInput


def myReLU(x):
	return x * (x > 0).double() 