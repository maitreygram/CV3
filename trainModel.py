import sys
import os
import torchfile
import torch
import numpy as np
import Model
import Criterion
from Linear import Linear
from Relu import ReLu

def normalize(trainingDataSet):
	return (trainingDataSet - np.mean(trainingDataSet))/np.std(trainingDataSet)

def default(str):
	return str + ' [Default: %default]'

USAGE_STRING = ""

def readCommand( argv ):
	"Processes the command used to run from the command line."
	from optparse import OptionParser
	parser = OptionParser(USAGE_STRING)

	parser.add_option('-m', '--modelName', help=default('modelConfig'))
	parser.add_option('-d', '--data', help=default('input'), default='data.bin', type="string")
	parser.add_option('-t', '--target', help=default('gradoutput'), default='labels.bin', type="string")

	options, otherjunk = parser.parse_args(argv)
	if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
	args = {}

	model_name = options.modelName
	training_data_path = options.data
	target_labels_path = options.target

	Data = torchfile.load(training_data_path)
	Labels = torchfile.load(target_labels_path)

	Data = torch.tensor(normalize(Data)).double()
	Data = Data.reshape(Data.shape[0],108*108)
	Labels = torch.tensor(Labels).long()

	# trainingData = Data[0:int(Data.shape[0]*0.9),:]
	# trainingLabels = Labels[0:int(Data.shape[0]*0.9)]

	my_model = Model.Model()
	my_model.addLayer(Linear(108*108,1024))
	my_model.addLayer(ReLu())
	my_model.addLayer(Linear(1024,256))
	my_model.addLayer(ReLu())
	my_model.addLayer(Linear(256,6))
	my_model.addLayer(ReLu())

	train_and_test(my_model, Data, Labels, 1, 432, 0.01, 0.001)
	try:
		os.mkdir(model_name)
	except:
		pass
	weights = []
	weights.append(my_model.layers[0].W)
	weights.append(my_model.layers[0].B)
	weights.append(my_model.layers[2].W)
	weights.append(my_model.layers[2].B)
	weights.append(my_model.layers[4].W)
	weights.append(my_model.layers[4].B)
	torch.save(weights, model_name + "/model.bin")


def train_and_test(my_model, trainingData, trainingLabels, noIters, batchSize, alpha, lr): # can add lambda
	# best_accuracy = 0
	# best_epoch = 0
	noBatches = int(trainingLabels.shape[0]/batchSize)
	my_criterion = Criterion.Criterion()

	# loss = np.zeros(noBatches)
	# accuracy = np.zeros(noBatches)

	for iter in range(noIters):
		for batch in range(noBatches):
			trData = trainingData[batch*batchSize:(batch+1)*batchSize,:]
			trLabels = trainingLabels[batch*batchSize:(batch+1)*batchSize]

			trOutputs = my_model.forward(trData, True)
			# loss_tr = my_criterion.forward(trOutputs, trLabels).item()

			dl_do = my_criterion.backward(trOutputs, trLabels)
			my_model.backward(trData, dl_do, alpha, lr)
			trOutputs = trOutputs.argmax(1)
			
			# accuracy[batch] = getAccuracy(trOutputs, trLabels)
			# loss[batch] = loss_tr
			# print("epoch: ", iter," [",batch,"/",noBatches,"]", "  training accuracy: ", accuracy[batch], "%  training Loss: ", loss_tr)

		# trAccuracy = np.mean(accuracy)
		# loss_tr = np.mean(loss)

		# valOutputs = my_model.forward(validationData, True)
		# loss_val = my_criterion.forward(valOutputs, validationLabels).item()
		# valOutputs = valOutputs.argmax(1)
		# valAccuracy = getAccuracy(valOutputs, validationLabels)

		# print("epoch: ", iter, "  training accuracy: ", trAccuracy, "%  training Loss: ", loss_tr, "  validation accuracy: ", valAccuracy, "%  validation Loss: ", loss_val)

readCommand(sys.argv[1:])