import torch 
import torchfile
import Model
import Criterion
import numpy as np
from Linear import Linear
from Relu import ReLu
import sys

def default(str):
	return str + ' [Default: %default]'

USAGE_STRING = ""

def readCommand( argv ):
	"Processes the command used to run from the command line."
	from optparse import OptionParser
	parser = OptionParser(USAGE_STRING)

	parser.add_option('-c', '--config', help=default('modelConfig'), default='CS 763 Deep Learning HW/modelConfig_1.txt')
	parser.add_option('-i', '--i', help=default('input'), default='CS 763 Deep Learning HW/input_sample_1.bin', type="string")
	parser.add_option('-g', '--og', help=default('gradoutput'), default='CS 763 Deep Learning HW/gradOutput_sample_1.bin', type="string")
	parser.add_option('-o', '--o', help=default('output'), type="string")
	parser.add_option('-w', '--ow', help=default('gradweights'), type="string")
	parser.add_option('-b', '--ob', help=default('gradb'), type="string")
	parser.add_option('-d', '--ig', help=default('gradinput'), type="string")

	options, otherjunk = parser.parse_args(argv)
	if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
	args = {}

	model_config_path = options.config
	input_path = options.i
	gradoutput_path = options.og
	output_path = options.o
	gradweights_path = options.ow
	gradb_path = options.ob
	gradinput_path = options.ig

	modelConfig_file = open(model_config_path, "r")
	data = modelConfig_file.readlines()

	my_model = Model.Model()
	my_criterion = Criterion.Criterion()
	
	input_weight = 0
	Bias_weight = 0

	Number_layer = int(data[0])
	for i in range (Number_layer):
		layer = data[1+i].split()
		if (len(layer) > 1):
			my_model.addLayer(Linear(int(layer[1]),int(layer[2])))
		else:
			my_model.addLayer(ReLu())

	Path_sample_weight = data[Number_layer+1][:-1]
	Path_sample_bias = data[Number_layer+2][:-1]

	input = torchfile.load(input_path)
	input = torch.tensor(input).double().reshape((input.shape[0],-1))

	input_weight = torchfile.load(Path_sample_weight)
	input_bias = torchfile.load(Path_sample_bias)

	input_weight = [torch.tensor(weight).double() for weight in input_weight]
	input_bias = [torch.tensor(bias).double().reshape((-1,1)) for bias in input_bias]

	Outputs = my_model.forward2(input, input_weight, input_bias, True)
	dl_do = my_criterion.backward(Outputs, trLabels)
	# gradoutput = my_model.backward(input, dl_do, 0, 0)

	[gradInput, gradWeights, gradBias] = my_model.backward2(input, dl_do, 0, 0)

	torch.save(Outputs, output_path)
	torch.save(gradWeights, gradweights_path)
	torch.save(gradBias, gradb_path)
	torch.save(gradInput, gradinput_path)

readCommand(sys.argv[1:])