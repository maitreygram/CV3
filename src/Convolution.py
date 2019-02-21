import numpy as np
import torch
import torchfile
import math

class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride):
		
		self.in_depth, self.in_row, self.in_col = in_channels
		# self.filter_row, self.filter_col = filter_size
		self.stride = stride
		self.filter_size = filter_size[0]
		self.out_depth = numfilters
		self.out_row = int((in_channels[1] - filter_size[0])/self.stride + 1)
		self.out_col = int((in_channels[2] - filter_size[0])/self.stride + 1)
		self.output = None
		# Stores the outgoing summation of weights * feautres 
		self.data = None
		self.W = torch.randn(numfilters, in_channels[0], filter_size[0], filter_size[0], dtype = torch.double)
		self.B = torch.randn(self.out_depth, dtype = torch.double)/math.sqrt(in_channels[0]*in_channels[1]*in_channels[2])
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		# self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		# self.biases = np.random.normal(0,0.1,self.out_depth)
		self.gradW = torch.empty(self.W.shape, dtype = torch.double)
		self.gradB = torch.empty(self.B.shape, dtype = torch.double)
		self.gradInput = None

	def forwardpass(self, X):
		# print('Forward CN ',self.weights.shape)
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]
		self.output = torch.empty(X.shape[0], self.out_depth, self.out_row, self.out_col, dtype = torch.double)
		###############################################
		# TASK 1 - YOUR CODE HERE
		# print("Conv layer")
		# print(X.shape)
		# print(self.stride)
		# result = np.zeros((n, self.out_depth, self.out_row, self.out_col))
		# print(X[0,0,0,0])
		# print(X.shape, self.weights.shape, self.filter_row, self.biases.shape)
		# print(self.in_depth, self.in_row, self.in_col)
		print X.shape
		for i in range(self.out_depth):
			for j in range(self.out_row):
				for k in range(self.out_col):
					# a = np.array(X[l,:,(j*self.stride):(j*self.stride+self.filter_row),(k*self.stride):(k*self.stride+self.filter_col)])
					# b = np.array(self.weights[i,:,:,:])
					# print((a*b).sum())
					self.output[:,i,j,k] = (self.W[i] * X[:,:,(j*self.stride):(j*self.stride+self.filter_size),(k*self.stride):(k*self.stride+self.filter_size)]).sum(1).sum(1).sum(1)
					# result[l,i,j,k] = (a*b).sum() + self.biases[i]
			self.output[:,i,:,:] += self.B[i]
		# self.data = result 
		return self.output
		
		# raise NotImplementedError
		###############################################

	def backwardpass(self, input, delta, alpha):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = input.shape[0] # batch size

		# mydelta = delta * derivative_sigmoid(self.data)
		# bias_grad = np.zeros(self.biases.shape)
		# w_grad = np.zeros(self.weights.shape)

		for i in range(self.W.shape[0]):
			for j in range(self.weights.shape[1]):
				for k in range(self.weights.shape[2]):
					for l in range(self.weights.shape[3]):
						# a = helper_funct(activation_prev[:,j],k,l,self.stride,self.out_row, self.out_col)
						# helper_delta = delta[:,i] * a
						# w_grad[i,j,k,l] = helper_delta.sum()
						self.gradW[i,j,k,l] = torch.mul(delta[:,i], input[:,j][:,k:self.out_row*self.stride+k:self.stride,l:l+self.stride*self.out_col:self.stride]).sum()
			self.gradB[i] = delta[:,i].sum()

		# return_delta = np.zeros(activation_prev.shape)
		self.gradInput = torch.zeros(input.shape, dtype = torch.double)
		reshaped_weight = self.W.reshape([1,self.W.shape[0],self.W.shape[1], self.W.shape[2], self.W.shape[3]])

		for i in range(self.out_row):
			for j in range(self.out_col):
				# return_delta[i,:,k*self.stride:k*self.stride+self.filter_row,l*self.stride:l*self.stride+self.filter_col] += self.weights[j] * mydelta[i,j,k,l]
				self.gradInput[:,:,i*self.stride:i*self.stride+self.filter_size,j*self.stride:j*self.stride+self.filter_size] += torch.mul(reshaped_weight,delta[:,:,i,j].reshape(n,self.out_depth,1,1,1)).sum(1)
		self.W -= alpha * self.gradW

		self.B -= alpha * self.gradB

		return self.gradInput
		###############################################
		# TASK 2 - YOUR CODE HERE
		# raise NotImplementedError