import torch
import numpy
class Model:
	def __init__(self):
		self.layers = []
		self.isTrain = False

	def forward(self, input, isTrain):
		output = input.clone()
		for l in self.layers:
			output = l.forwardpass(output)
		return output

	def forward2(self, input1, W, B, isTrain):
		output = input1.clone()
		for l in range(len(self.layers)):
			output = self.layers[l].forwardpass2(output, W[l], B[l])
		return output

	def backward(self, input, gradOutput, alpha, learning_rate):
		self.clearGradParam(input.shape)
		for i in range(len(self.layers)-1, 0, -1):
			gradOutput = self.layers[i].backwardpass(self.layers[i-1].output, gradOutput, alpha, learning_rate)

		gradOutput = self.layers[0].backwardpass(input, gradOutput, alpha, learning_rate)
		return gradOutput

	def backward2(self, input, gradOutput, alpha, learning_rate):
		self.clearGradParam(input.shape)
		gradWeights = []
		gradInput = []
		gradBias = []
		for i in range(len(self.layers)-1, 0, -1):
			[gradOutput, gradW, gradB] = self.layers[i].backwardpass2(self.layers[i-1].output, gradOutput, alpha, learning_rate)
			gradInput.append(gradOutput)
			if gradW != None:
				gradWeights.append(gradW)
				gradBias.append(gradB)

		[gradOutput, gradW, gradI] = self.layers[0].backwardpass2(input, gradOutput, alpha, learning_rate)
		gradInput.append(gradOutput)
		if gradW != None:
			gradWeights.append(gradW)
			gradBias.append(gradB)
		return gradInput, gradWeights, gradBias

	def dispGradParam(self):
		for i in range(len(self.layers)-1, 0, -1):
			print("Grad Input for layer ", i, " is ", self.layers[i].gradInput)

	def clearGradParam(self, inputShape):
		for i in range(len(self.layers)-1, 0, -1):
			self.layers[i].gradInput = torch.zeros(inputShape, dtype = torch.double)

	def addLayer(self, layer):
		self.layers.append(layer)

	def retWeights(self):
		weights = []
		for l in self.layers:
			if l.type == "Linear":
				weights.append(l.W)

		return weights