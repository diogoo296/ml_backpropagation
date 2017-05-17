#!/usr/bin/env python
import numpy as np
import random
# import sys


class NeuralNetwork:
    NUM_INPUTS = 784
    NUM_OUTPUTS = 10

    def __init__(self, numHiddenLayers, alpha, nEpochs):
        sizes = [self.NUM_INPUTS, numHiddenLayers, self.NUM_OUTPUTS]

        self.num_layers = len(sizes)
        self.b = [np.random.randn(y, 1) for y in sizes[1:]]
        self.W = [np.random.randn(y, x) / np.sqrt(x)
                  for x, y in zip(sizes[:-1], sizes[1:])]

        self.alpha = alpha
        self.nEpochs = nEpochs

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forwardPass(self, a):
        for b, w in zip(self.b, self.W):
            a = self.sigmoid(np.dot(w, a) + b)

        return a

    def crossEntropy(self, y, a):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    def softmax(self, output):
        return np.exp(output) / float(sum(np.exp(output)))

    def update_mini_batch(self, mini_batch, n):
        sumGradsB = [np.zeros(b.shape) for b in self.b]
        sumGradsW = [np.zeros(w.shape) for w in self.W]

        for x in mini_batch:
            gradsB, gradsW = self.backprop(x[:-1], x[-1])
            sumGradsB = [sumB + gradB for sumB, gradB in zip(sumGradsB, gradsB)]
            sumGradsW = [sumW + gradW for sumW, gradW in zip(sumGradsW, gradsW)]

        self.W = [
            # (1 - self.alpha / n) * w - (self.alpha / len(mini_batch)) * sumW
            w + (self.alpha * sumW / len(mini_batch))
            for w, sumW in zip(self.W, sumGradsW)
        ]
        self.b = [
            # b - (self.alpha / len(mini_batch)) * sumB
            b + (self.alpha * sumB / len(mini_batch))
            for b, sumB in zip(self.b, sumGradsB)
        ]

    def derivative(self, z, a, y):
        return (a - y) * self.sigmoid_prime(z)

    def calculateLoss(self, output, expected):
        encoded = np.zeros((self.NUM_OUTPUTS, 1))
        encoded[int(expected)] = 1.0

        self.cost += self.crossEntropy(encoded, output)

    def predict(self, output, y):
        if np.argmax(self.softmax(output)) == y:
            self.correct += 1

    def backprop(self, x, y):
        gradsB = [np.zeros(b.shape) for b in self.b]
        gradsW = [np.zeros(w.shape) for w in self.W]

        # feedforward
        x = x.reshape(self.NUM_INPUTS, 1)
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer

        for b, w in zip(self.b, self.W):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        self.calculateLoss(activation, y)
        self.predict(activation, y)

        # backward pass
        delta = self.derivative(zs[-1], activations[-1], y)
        gradsB[-1] = delta
        gradsW[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.W[-l+1].transpose(), delta) * sp
            gradsB[-l] = delta
            gradsW[-l] = np.dot(delta, activations[-l-1].transpose())
        # for l in reversed(range(len(zs) - 2)):
        #     z = zs[l]
        #     sp = self.sigmoid_prime(z)
        #     delta = np.dot(self.W[l].transpose(), delta) * sp
        #     gradsB[l] = delta
        #     gradsW[l] = np.dot(delta, activations[l-1].transpose())

        return (gradsB, gradsW)

    def gradientDescent(self, trainingSet, mini_batch_size):
        trainingCost = []
        training_data = trainingSet.data

        for j in range(self.nEpochs):
            random.shuffle(training_data)
            self.cost = 0.0
            self.correct = 0
            mini_batches = [
                training_data[k: k + mini_batch_size]
                for k in range(0, trainingSet.size, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, len(training_data))

            cost = self.cost / trainingSet.size
            trainingCost.append(cost)
            # print('> epoch=%d, error=%.3f' % (j, cost))
            print('> epoch=%d, error=%.3f' % (j, cost))
            print('< correct=%d/%d' % (self.correct, trainingSet.size))
