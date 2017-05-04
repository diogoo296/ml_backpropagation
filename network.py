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

    # Calculate the derivative of an neuron output
    def transferDerivative(self, output):
        return output * (1.0 - output)

    # Backpropagate error and store in neurons
    def backPropagate(self, expected):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            errors = list()
            if i != len(self.layers) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.layers[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])

                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])

            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = (
                    errors[j] * self.transferDerivative(neuron['output'])
                )

# test backpropagation
seed(1)
neuralNet = NeuralNetwork(1, 2, 2)
neuralNet.layers = [
    [{'output': 0.7105668883115941, 'weights': [
        0.13436424411240122, 0.8474337369372327, 0.763774618976614
    ]}],
    [
        {'output': 0.6213859615555266, 'weights': [
            0.2550690257394217, 0.49543508709194095
        ]},
        {'output': 0.6573693455986976, 'weights': [
            0.4494910647887381, 0.651592972722763
        ]}
    ]
]
expected = [0, 1]
neuralNet.backPropagate(expected)
for layer in neuralNet.layers:
    print(layer)
