import numpy as np
import torchfile
import torch
# data = torchfile.load("data.bin")

class Linear:
	def __init__(self, in_nodes, out_nodes):
		# self.in_nodes = in_nodes
		# self.out_nodes = out_nodes
		# self.data = None
		batch = 1
		self.output = None
		# self.W = torch.empty(out_nodes, in_nodes)
		self.W = torch.randn(out_nodes, in_nodes, dtype=torch.double) * 0.01
		self.B = torch.randn(out_nodes, 1, dtype=torch.double) * 0.01
		self.gradW = torch.empty(out_nodes, in_nodes, dtype=torch.double)
		self.gradB = torch.empty(out_nodes, 1, dtype=torch.double)
		self.gradInput = None

	def forwardpass(self, input):	
		self.output = torch.mm(input, self.W.t()) + self.B.t()
		return self.output

	def backwardpass(self, input, gradOutput, learning_rate):
		self.gradW = torch.mm(gradOutput.t(), input)
		self.gradInput = torch.mm(gradOutput, self.W)
		self.gradB = sum(gradOutput).reshape(self.B.shape)

		self.W -= learning_rate * self.gradW
		self.B -= learning_rate * self.gradB
		return self.gradInput
