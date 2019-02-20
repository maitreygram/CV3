import Model
import Criterion
import torch
import torchfile
from Linear import Linear
from Relu import ReLu

def getAccuracy(I, J):
	return (torch.sum((I == J)).double()/float(I.shape[0])).item() * 100.0

def train(Data, Labels, noIters, batchSize, lr): # can add lambda
	best_accuracy = 0
	best_epoch = 0
	for iter in range(noIters):
		# if iter % 20 == 0:
			# lr = lr/10
		outputs = my_model.forward(Data, True)
		loss_tr = my_criterion.forward(outputs, Labels)
		dl_do = my_criterion.backward(outputs, Labels)
		my_model.backward(Data, dl_do, lr)

		outputs = outputs.argmax(1)
		accuracy = getAccuracy(outputs, Labels)

		if accuracy > best_accuracy:
			best_accuracy = accuracy
			best_epoch = iter

		print("epoch: ", iter, "  accuracy: ", accuracy, "%  Loss: ", loss_tr)
	print("best accuracy: ", best_accuracy, "  on epoch: ", best_epoch)



trainingData = torch.tensor(torchfile.load("data.bin")).double()
trainingLabels = torch.tensor(torchfile.load("labels.bin")).double()

my_model = Model.Model()
my_model.addLayer(Linear(108*108,6))
# my_model.addLayer(ReLu())
# my_model.addLayer(Linear(1024,256))
# my_model.addLayer(ReLu())
# my_model.addLayer(Linear(256,6))
# my_model.addLayer(ReLu())

my_criterion = Criterion.Criterion()

trainingDataSet = trainingData[0:100]
trainingLabelsSet = trainingLabels[0:100]

trainingDataSet = (trainingDataSet - torch.mean(trainingDataSet))/255.0
trainingLabelsSet = trainingLabelsSet.long()

# print(trainingLabelsSet)

trainingDataSet = trainingDataSet.reshape(trainingDataSet.shape[0],108*108)
# trainingDataSet = torch.from_numpy(trainingDataSet).double()
# trainingLabelsSet = trainingLabelsSet.reshape(trainingLabelsSet.shape[0],1)
# trainingLabelsSet = torch.from_numpy(trainingLabelsSet).double()

train(trainingDataSet, trainingLabelsSet, 1000, 100, 0.00001)
# output = my_model.forward(trainingDataSet, True)
# print(output.shape)
# loss_tr = my_criterion.forward(output, trainingLabelsSet)
# print("loss_tr ", loss_tr)
