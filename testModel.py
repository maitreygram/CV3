import sys
import os
import torchfile
import torch
import Model
import Criterion
import torch
import torchfile
import numpy as np
from Linear import Linear
from Relu import ReLu
import csv

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
	parser.add_option('-t', '--test', help=default('input'), default='test.bin', type="string")

	options, otherjunk = parser.parse_args(argv)
	if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
	args = {}

	model_name = options.modelName
	test_data_path = options.test
	# target_labels_path = options.target

	Data = torchfile.load(test_data_path)

	Data = torch.tensor(normalize(Data)).double()
	print(Data.shape)
	Data = Data.reshape(Data.shape[0],108*108)

	my_model = Model.Model()
	my_model.addLayer(Linear(108*108,1024))
	my_model.addLayer(ReLu())
	my_model.addLayer(Linear(1024,256))
	my_model.addLayer(ReLu())
	my_model.addLayer(Linear(256,6))
	my_model.addLayer(ReLu())

	model = torch.load(model_name)
	my_model.layers[0].W = model[0]
	my_model.layers[0].B = model[1]
	my_model.layers[2].W = model[2]
	my_model.layers[2].B = model[3]
	my_model.layers[4].W = model[4]
	my_model.layers[4].B = model[5]

	train_and_test(my_model, Data, 1, 432, 0.01, 0.001)


def train_and_test(my_model, trainingData, noIters, batchSize, alpha, lr): # can add lambda
	# best_accuracy = 0
	# best_epoch = 0
	# global my_model
	noBatches = int(trainingData.shape[0]/batchSize)

	# loss = np.zeros(noBatches)
	# accuracy = np.zeros(noBatches)
	labels = torch.tensor([]).long()

	for batch in range(noBatches):
		trData = trainingData[batch*batchSize:(batch+1)*batchSize,:]
		# trLabels = trainingLabels[batch*batchSize:(batch+1)*batchSize]

		trOutputs = my_model.forward(trData, True)
		# loss_tr = my_criterion.forward(trOutputs, trLabels).item()

		# dl_do = my_criterion.backward(trOutputs, trLabels)
		# my_model.backward(trData, dl_do, alpha, lr)
		trOutputs = trOutputs.argmax(1)
		labels = torch.cat((labels, trOutputs),0)
		# print(labels.shape)
			# accuracy[batch] = getAccuracy(trOutputs, trLabels)
			# loss[batch] = loss_tr
			# print("epoch: ", iter," [",batch,"/",noBatches,"]", "  training accuracy: ", accuracy[batch], "%  training Loss: ", loss_tr)

	trData = trainingData[noBatches*batchSize:trainingData.shape[0],:]
	trOutputs = my_model.forward(trData, True)
	trOutputs = trOutputs.argmax(1)
	labels = torch.cat((labels, trOutputs),0)
	
	csvFile = []
	for i in range(0, trainingData.shape[0]):
		row = ([str(i), str(labels[i].item())])
		csvFile.append(row)
		# print(row)
		# trAccuracy = np.mean(accuracy)
		# loss_tr = np.mean(loss)
	# print(csvFile)

	with open('testPrediction.bin', 'w') as f:
		writer = csv.writer(f)
		writer.writerows(csvFile)

	f.close()
		# valOutputs = my_model.forward(validationData, True)
		# loss_val = my_criterion.forward(valOutputs, validationLabels).item()
		# valOutputs = valOutputs.argmax(1)
		# valAccuracy = getAccuracy(valOutputs, validationLabels)

		# print("epoch: ", iter, "  training accuracy: ", trAccuracy, "%  training Loss: ", loss_tr, "  validation accuracy: ", valAccuracy, "%  validation Loss: ", loss_val)

readCommand(sys.argv[1:])