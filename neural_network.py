#!/usr/bin/env python
from random import random, seed
from math import exp


class NeuralNetwork:
    NUM_INPUTS = 784
    NUM_OUTPUTS = 10

    def __init__(self, numHiddenLayers, learningRate, numEpochs):
        self.layers = list()
        self.learningRate = learningRate
        self.numEpochs = numEpochs

        hiddenLayer = [
            {'weights': [random() for i in range(self.NUM_INPUTS + 1)]}
            for i in range(numHiddenLayers)
        ]
        self.layers.append(hiddenLayer)

        outputLayer = [
            {'weights': [random() for i in range(numHiddenLayers + 1)]}
            for i in range(self.NUM_OUTPUTS)
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

    # Update network weights with error
    def updateWeights(self, row):
        for i in range(len(self.layers)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.layers[i - 1]]

            for neuron in self.layers[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += (
                        self.learningRate * neuron['delta'] * inputs[j]
                    )

                neuron['weights'][-1] += self.learningRate * neuron['delta']

    # Train a network for a fixed number of epochs
    def train(self, trainingSet):
        for epoch in range(self.numEpochs):
            errorSum = 0
            for row in trainingSet:
                outputs = self.forwardPropagate(row)
                expected = [0 for i in range(self.NUM_OUTPUTS)]
                expected[row[-1]] = 1
                errorSum += sum([
                    (expected[i] - outputs[i])**2 for i in range(len(expected))
                ])
                self.backPropagate(expected)
                self.updateWeights(row)

            print('>epoch=%d, lrate=%.3f, error=%.3f' % (
                epoch, self.learningRate, errorSum
            ))
