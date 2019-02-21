import Model
import Criterion
import torch
import torchfile
import numpy as np
from Linear import Linear
from Relu import ReLu

def normalize(trainingDataSet):
	return (trainingDataSet - np.mean(trainingDataSet))/np.std(trainingDataSet)

def getAccuracy(I, J):
	return (torch.sum((I == J)).double()/float(I.shape[0])).item() * 100.0

Data = torchfile.load("data.bin")
Labels = torchfile.load("labels.bin")

Data = torch.tensor(normalize(Data)).double()
Data = Data.reshape(Data.shape[0],108*108)
Labels = torch.tensor(Labels).long()

trainingData = Data[0:int(Data.shape[0]*0.9),:]
trainingLabels = Labels[0:int(Data.shape[0]*0.9)]

validationData = Data[int(Data.shape[0]*0.9):Data.shape[0],:]
validationLabels = Labels[int(Data.shape[0]*0.9):Data.shape[0]]

my_model = Model.Model()
my_model.addLayer(Linear(108*108,1024))
my_model.addLayer(ReLu())
my_model.addLayer(Linear(1024,256))
my_model.addLayer(ReLu())
my_model.addLayer(Linear(256,6))
my_model.addLayer(ReLu())

my_criterion = Criterion.Criterion()

def train_and_test(trainingData, trainingLabels, validationData, validationLabels, noIters, batchSize, alpha, lr): # can add lambda
	global my_model
	noBatches = int(trainingLabels.shape[0]/batchSize)

	loss = np.zeros(noBatches)
	accuracy = np.zeros(noBatches)

	for iter in range(noIters):
		for batch in range(noBatches):
			trData = trainingData[batch*batchSize:(batch+1)*batchSize,:]
			trLabels = trainingLabels[batch*batchSize:(batch+1)*batchSize]

			trOutputs = my_model.forward(trData, True)
			loss_tr = my_criterion.forward(trOutputs, trLabels).item()

			dl_do = my_criterion.backward(trOutputs, trLabels)
			my_model.backward(trData, dl_do, alpha, lr)
			trOutputs = trOutputs.argmax(1)
			
			accuracy[batch] = getAccuracy(trOutputs, trLabels)
			loss[batch] = loss_tr
			# print("epoch: ", iter," [",batch,"/",noBatches,"]", "  training accuracy: ", accuracy[batch], "%  training Loss: ", loss_tr)

		trAccuracy = np.mean(accuracy)
		loss_tr = np.mean(loss)

		valOutputs = my_model.forward(validationData, True)
		loss_val = my_criterion.forward(valOutputs, validationLabels).item()
		valOutputs = valOutputs.argmax(1)
		valAccuracy = getAccuracy(valOutputs, validationLabels)

		print("epoch: ", iter, "  training accuracy: ", trAccuracy, "%  training Loss: ", loss_tr, "  validation accuracy: ", valAccuracy, "%  validation Loss: ", loss_val)
		# if accuracy > best_accuracy:
			# best_accuracy = accuracy
			# best_epoch = iter

	# print("best accuracy: ", best_accuracy, "  on epoch: ", best_epoch)

train_and_test(trainingData, trainingLabels, validationData, validationLabels, 300, 432, 0.01, 0.001)

weights = []
weights.append(my_model.layers[0].W)
weights.append(my_model.layers[0].B)
weights.append(my_model.layers[2].W)
weights.append(my_model.layers[2].B)
weights.append(my_model.layers[4].W)
weights.append(my_model.layers[4].B)
torch.save(weights, "Model.bin")

# trainingDataSet = trainingData[0:100]
# trainingLabelsSet = trainingLabels[0:100]

# trainingLabelsSet = trainingLabelsSet.long()

# print(trainingLabelsSet)

# trainingDataSet = trainingDataSet.reshape(trainingDataSet.shape[0],108*108)
# trainingDataSet = torch.from_numpy(trainingDataSet).double()
# trainingLabelsSet = trainingLabelsSet.reshape(trainingLabelsSet.shape[0],1)
# trainingLabelsSet = torch.from_numpy(trainingLabelsSet).double()


# trained_model = my_model.retWeights()

# output = my_model.forward(trainingDataSet, True)
# print(output.shape)
# loss_tr = my_criterion.forward(output, trainingLabelsSet)
# print("loss_tr ", loss_tr)


# save best model
