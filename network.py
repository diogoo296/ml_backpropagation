#!/usr/bin/env python
from random import random, seed
from math import exp


class NeuralNetwork:
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

    def activate(self, weights, inputs):
        activation = weights[-1]  # Bias
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]

        return activation

    # Transfer neuron activation using the sigmoid function
    def transfer(self, activation):
        return 1.0 / (1.0 + exp(-activation))

    # Forward propagate input to a network output
    def forwardPropagate(self, row):
        inputs = row
        for layer in self.layers:
            newInputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                newInputs.append(neuron['output'])

            inputs = newInputs

        return inputs

# test forward propagation
seed(1)
neuralNet = NeuralNetwork(1, 2, 2)
neuralNet.layers = [
    [{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
    [
        {'weights': [0.2550690257394217, 0.49543508709194095]},
        {'weights': [0.4494910647887381, 0.651592972722763]}
    ]
]
row = [1, 0, None]
print(neuralNet.forwardPropagate(row))
