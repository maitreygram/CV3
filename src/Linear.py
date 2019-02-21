import numpy as np
import torchfile
import torch
import math

class Linear:
	def __init__(self, in_nodes, out_nodes):
		# self.in_nodes = in_nodes
		# self.out_nodes = out_nodes
		# self.data = None
		self.type = "Linear"
		self.output = None
		# output = None
		# self.W = torch.empty(out_nodes, in_nodes)
		self.W = torch.randn(out_nodes, in_nodes, dtype=torch.double)/math.sqrt(in_nodes)
		self.B = torch.randn(out_nodes, 1, dtype=torch.double)/math.sqrt(in_nodes)
		self.gradW = torch.empty(out_nodes, in_nodes, dtype=torch.double)
		self.gradB = torch.empty(out_nodes, 1, dtype=torch.double)
		self.Wm = torch.zeros(out_nodes, in_nodes, dtype=torch.double)
		self.Bm = torch.zeros(out_nodes, 1, dtype=torch.double)
		self.gradInput = None

	def forwardpass(self, input):	
		self.output = torch.mm(input, self.W.t()) + self.B.t()
		return self.output

	def forwardpass2(self, input1, W, B):	
		print(W.shape, input1.shape, B.shape)
		print(type(torch.mm(input1, W.t()) + B.t()))
		out = torch.mm(input1, W.t()) + B.t()
		print(type(out))
		return out

	def backwardpass(self, input, gradOutput, alpha, learning_rate):
		self.gradW = torch.mm(gradOutput.t(), input)
		self.gradInput = torch.mm(gradOutput, self.W)
		self.gradB = sum(gradOutput).reshape(self.B.shape)

		self.Wm = alpha * self.Wm - learning_rate * self.gradW
		self.Bm = alpha * self.Bm - learning_rate * self.gradB
		self.W += self.Wm
		self.B += self.Bm
		return self.gradInput

	def backwardpass2(self, input, gradOutput, alpha, learning_rate):
		self.gradW = torch.mm(gradOutput.t(), input)
		self.gradInput = torch.mm(gradOutput, self.W)
		self.gradB = sum(gradOutput).reshape(self.B.shape)

		self.Wm = alpha * self.Wm - learning_rate * self.gradW
		self.Bm = alpha * self.Bm - learning_rate * self.gradB
		self.W += self.Wm
		self.B += self.Bm
		return [self.gradInput, self.gradW, self.gradB]
