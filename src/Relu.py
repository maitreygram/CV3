import torch

class ReLu:
	def __init__(self):
		# batch = 1
		self.type = "ReLu"
		self.output = None
		self.gradInput = None

	def forwardpass(self, input):
		self.output = myReLU(input)
		return self.output

	def backwardpass(self, input, gradOutput, alpha, learning_rate):
		self.gradInput = gradOutput * (input > 0).double()
		return self.gradInput

	def backwardpass2(self, input, gradOutput, alpha, learning_rate):
		self.gradInput = gradOutput * (input > 0).double()
		return self.gradInput, None, None


def myReLU(x):
	return x * (x > 0).double() 