#!/usr/bin/env python
from random import random
from math import exp, log
import numpy as np


class NeuralNetwork:
    NUM_INPUTS = 784
    NUM_OUTPUTS = 10

    def __init__(self, numHiddenLayers, lmbda, numEpochs):
        self.layers = list()
        self.lmbda = lmbda
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

    # Forward propagate input to network output
    def forwardPropagate(self, row):
        inputs = row
        for layer in self.layers:
            newInputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                newInputs.append(neuron['output'])

            inputs = newInputs

        self.outputs = inputs

    def softmax(self):
        self.outputs = np.exp(self.outputs) / float(sum(np.exp(self.outputs)))
        # self.outputs = (
        #    np.exp(self.outputs) / np.sum(np.exp(self.outputs), axis=0)
        # )
        # return result
        # for i in range(self.NUM_OUTPUTS):
        #    self.layers[len(self.layers) - 1][i]['output'] = result[i]

    def oneHotEncoding(self, expected):
        maxIdx = 9
        encoded = [0.0 for i in range(maxIdx + 1)]
        encoded[expected] = 1.0
        return encoded

    # Calculate the derivative of an neuron output
    def transferDerivative(self, output):
        return output * (1.0 - output)

    def calculateLoss(self, expected):
        # output = [l['output'] for l in self.layers[len(self.layers) - 1]]
        self.softmax()
        output = self.outputs
        error = 0.0
        for i in range(self.NUM_OUTPUTS):
            e1 = (-1.0) * expected[i] * log(output[i], 10)
            e2 = (1.0 - expected[i]) * log(1 - output[i], 10)
            error += e1 - e2

        return error

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
                # neuron['delta'] = errors[j]
                neuron['delta'] = (
                    errors[j] * self.transferDerivative(neuron['output'])
                )

    # Update network weights with error
    def updateWeights(self, inputs):
        for i in range(len(self.layers)):
            if i != 0:
                inputs = [neuron['output'] for neuron in self.layers[i - 1]]

            for neuron in self.layers[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += (
                        self.lmbda * neuron['delta'] * inputs[j]
                    )

                neuron['weights'][-1] += self.lmbda * neuron['delta']

    def predict(self):
        return np.argmax(self.outputs)

    # Train a network for a fixed number of epochs
    def sgd(self, trainingSet):
        for epoch in range(self.numEpochs):
            lossSum = 0
            correct = 0
            # for row in trainingSet:
            for k in range(trainingSet.size):
                row = trainingSet.inputs[k]
                expected = self.oneHotEncoding(trainingSet.outputs[k])

                self.forwardPropagate(row)

                lossSum += self.calculateLoss(expected)
                prediction = self.predict()

                if prediction == trainingSet.outputs[k]:
                    correct += 1
                # lossSum += sum(
                # [(expected[i]-outputs[i])**2 for i in range(len(expected))]
                # )
                self.backPropagate(expected)
                self.updateWeights(row)

            lossSum /= trainingSet.size

            print('> epoch=%d, lrate=%.3f, error=%.3f' % (
                epoch, self.lmbda, lossSum
            ))
            print('< score: %d/%d' % (correct, trainingSet.size))

    def gd(self, trainingSet):
        for epoch in range(self.numEpochs):
            lossSum = 0
            # for row in trainingSet:
            for k in range(trainingSet.size):
                row = trainingSet.inputs[k]
                expected = self.oneHotEncoding(trainingSet.outputs[k])

                self.forwardPropagate(row)
                lossSum += self.calculateLoss(expected)
                # lossSum += sum(
                # [(expected[i]-outputs[i])**2 for i in range(len(expected))]
                # )
                self.backPropagate(expected)

            lossSum /= trainingSet.size
            self.updateWeights(row)

            print('> epoch=%d, lrate=%.3f, error=%.3f' % (
                epoch, self.lmbda, lossSum
            ))
