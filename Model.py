import torch

class Model:
	def __init__(self):
		self.layers = []
		self.isTrain = False

	def forward(self, input, isTrain):
		output = input.clone()
		for l in self.layers:
			output = l.forwardpass(output)
		return output

	def backward(self, input, gradOutput, learning_rate):
		self.clearGradParam(input.shape)
		for i in range(len(self.layers)-1, 0, -1):
			gradOutput = self.layers[i].backwardpass(self.layers[i-1].output, gradOutput, learning_rate)

		gradOutput = self.layers[0].backwardpass(input, gradOutput, learning_rate)

	def dispGradParam(self):
		for i in range(len(self.layers)-1, 0, -1):
			print("Grad Input for layer ", i, " is ", self.layers[i].gradInput)

	def clearGradParam(self, inputShape):
		for i in range(len(self.layers)-1, 0, -1):
			self.layers[i].gradInput = torch.zeros(inputShape, dtype = torch.double)

	def addLayer(self, layer):
		self.layers.append(layer)