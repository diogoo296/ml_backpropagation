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

    # Update network weights with error
    def updateWeights(self, row, learningRate):
        for i in range(len(self.layers)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.layers[i - 1]]

            for neuron in self.layers[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += (
                        learningRate * neuron['delta'] * inputs[j]
                    )

                neuron['weights'][-1] += learningRate * neuron['delta']

    # Train a network for a fixed number of epochs
    def train(self, trainingSet, learningRate, numEpochs, numOutputs):
        for epoch in range(numEpochs):
            errorSum = 0
            for row in trainingSet:
                outputs = self.forwardPropagate(row)
                expected = [0 for i in range(numOutputs)]
                expected[row[-1]] = 1
                errorSum += sum([
                    (expected[i] - outputs[i])**2 for i in range(len(expected))
                ])
                self.backPropagate(expected)
                self.updateWeights(row, learningRate)

            print('>epoch=%d, lrate=%.3f, error=%.3f' % (
                epoch, learningRate, errorSum
            ))


# test backpropagation
seed(1)
dataset = [
    [2.7810836, 2.550537003, 0],
    [1.465489372, 2.362125076, 0],
    [3.396561688, 4.400293529, 0],
    [1.38807019, 1.850220317, 0],
    [3.06407232, 3.005305973, 0],
    [7.627531214, 2.759262235, 1],
    [5.332441248, 2.088626775, 1],
    [6.922596716, 1.77106367, 1],
    [8.675418651, -0.242068655, 1],
    [7.673756466, 3.508563011, 1]
]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
neuralNet = NeuralNetwork(n_inputs, 2, n_outputs)
neuralNet.train(dataset, 0.5, 1000, n_outputs)
for layer in neuralNet.layers:
    print(layer)
