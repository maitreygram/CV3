import sys
import torch
import torchfile
import Criterion

def default(str):
    return str + ' [Default: %default]'

USAGE_STRING = ""

def readCommand( argv ):
    "Processes the command used to run from the command line."
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-i', '--input_bin', help=default('The type of classifier'), default='CS_763_Deep_Learning_HW/input_criterion_sample_1.bin')
    parser.add_option('-t', '--target_bin', help=default('The size of the training set'), default='CS_763_Deep_Learning_HW/target_sample_1.bin', type="string")
    parser.add_option('-g', '--ig', help=default('Whether to use enhanced features'), default='CS_763_Deep_Learning_HW/gradCriterionInput_sample_1.bin', type="string")
    # parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
    # parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
    # parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
    # parser.add_option('-v', '--validate', help=default("Whether to validate when training (for graphs)"), default=False, action="store_true")
    # parser.add_option('-d', '--dataset', help=default("Specifies the data set to use"), choices=['d1', 'd2'], default='d1')
    # parser.add_option('-k', '--classes', help=default("Specifies the number of classes"), default=10, type="int")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    input_bin_path = options.input_bin
    target_bin_path = options.target_bin
    grad_input_bin_path = options.ig

    input_bin = torch.tensor(torchfile.load(input_bin_path)).double()
    target_bin = torch.tensor(torchfile.load(target_bin_path)).double()
    grad_input_bin = torch.tensor(torchfile.load(grad_input_bin_path)).double()

    size = target_bin.shape[0]
    # print size
    for j in range(size):
        target_bin[j] -= 1

    # print target_bin
    # grad_Input_bin = options.grad_Input_bin
    # print grad_input_bin
    # loss = Criterion.forward(input_bin, target_bin)
    # gradLoss = Criterion.backward(input_bin, target_bin)
    criterion = Criterion.Criterion()
    loss = criterion.forward(input_bin, target_bin)
    print "Loss is -----", loss
    # print loss
    # print input_bin.shape
    # print gradLoss
    gradInput = criterion.backward(input_bin, target_bin)


readCommand(sys.argv[1:])