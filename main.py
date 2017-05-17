#!/usr/bin/env python
import sys, getopt
from dataset import Dataset
from neural_network import NeuralNetwork


def main(argv):
    hiddenLayers = 0
    learningRate = 0
    epochs = 0
    inputFile = ''

    try:
        opts, args = getopt.getopt(argv,'hl:r:e:i:')
    except getopt.GetoptError:
        print('Usage: main.py -l <hiddenLayers> -r <learningRate> -e <epochs> -i <inputFile>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('Usage: main.py -l <hiddenLayers> -r <learningRate> -e <epochs> -i <inputFile>')
            sys.exit()
        elif opt == "-l":
            hiddenLayers = int(arg)
        elif opt == '-r':
            learningRate = float(arg)
        elif opt == '-e':
            epochs = int(arg)
        elif opt == '-i':
            inputFile = arg

    dataset = Dataset(inputFile)
    nn = NeuralNetwork(hiddenLayers, learningRate, epochs)
    nn.gradientDescent(dataset, 50)

if __name__ == "__main__":
    main(sys.argv[1:])
