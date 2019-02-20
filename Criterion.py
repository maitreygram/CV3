import torch

class Criterion:
	def __init__(self):
		pass

	def forward(self, input, target): # can add lambda
		probabilities = torch.exp(input)
		sum_batch = probabilities.sum(1)
		# print("Criterion forward: ", torch.mean(torch.log(sum_batch) - input[:,target.long()]))
		loss = torch.mean(torch.log(sum_batch) - input[:,target.long()])
		return loss

	def backward(self, input, target):
		probabilities = torch.exp(input)
		sum_batch = probabilities.sum(1)

		p = probabilities / sum_batch.reshape(input.shape[0],1)

		for j in range(input.shape[0]):
			p[j][target[j].long()] -= 1.0

		return p