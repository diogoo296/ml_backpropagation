#!/usr/bin/env python
from random import seed
from random import random


class Network:
    def __init__(self, numInputs, numHiddenLayers, numOutputs):
        self.layers = list()

        hiddenLayer = [
            {'weights': [random() for i in range(numInputs + 1)]}
            for i in range(numHiddenLayers)
        ]
        self.layers.append(hiddenLayer)

        outputLayer = [
            {'weights': [random() for i in range(numHiddenLayers + 1)]}
            for i in range(numOutputs)
        ]
        self.layers.append(outputLayer)

seed(1)
network = Network(2, 1, 2)
for layer in network.layers:
    print(layer)
