#!/usr/bin/env python
import numpy as np
import random


class NeuralNetwork:
    NUM_INPUTS = 784
    NUM_OUTPUTS = 10
    NUM_LAYERS = 3

    def __init__(self, numHiddenLayers, alpha, nEpochs):
        sizes = [self.NUM_INPUTS, numHiddenLayers, self.NUM_OUTPUTS]

        self.b = [np.random.randn(y, 1) for y in sizes[1:]]
        self.W = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]

        self.alpha = alpha
        self.nEpochs = nEpochs

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoidPrime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forwardPass(self, a):
        for b, w in zip(self.b, self.W):
            a = self.sigmoid(np.dot(w, a) + b)

        return a

    def crossEntropyLoss(self, y, a):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    def updateMiniBatch(self, mini_batch, n):
        sumGradsB = [np.zeros(b.shape) for b in self.b]
        sumGradsW = [np.zeros(w.shape) for w in self.W]

        for x in mini_batch:
            gradsB, gradsW = self.backpropagation(x[:-1], x[-1])
            sumGradsB = [sumB + gradB for sumB, gradB in zip(sumGradsB, gradsB)]
            sumGradsW = [sumW + gradW for sumW, gradW in zip(sumGradsW, gradsW)]

        self.W = [w - (self.alpha * sumW / len(mini_batch)) for w, sumW in zip(self.W, sumGradsW)]
        self.b = [b - (self.alpha * sumB / len(mini_batch)) for b, sumB in zip(self.b, sumGradsB)]

    def derivative(self, z, a, y):
        return (a - y) * self.sigmoidPrime(z)

    def oneHotEncoding(self, y):
        encoded = np.zeros((self.NUM_OUTPUTS, 1))
        encoded[int(y)] = 1.0
        return encoded

    def predict(self, output, y):
        softmax = np.exp(output) / float(sum(np.exp(output)))
        if np.argmax(softmax) == np.argmax(y):
            self.correct += 1

    def backpropagation(self, x, y):
        gradsB = [np.zeros(b.shape) for b in self.b]
        gradsW = [np.zeros(w.shape) for w in self.W]

        x = x.reshape(self.NUM_INPUTS, 1)
        y = self.oneHotEncoding(y)

        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer

        for b, w in zip(self.b, self.W):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        self.cost += self.crossEntropyLoss(y, activation)
        self.predict(activation, y)

        # backward pass
        delta = self.derivative(zs[-1], activations[-1], y)
        gradsB[-1] = delta
        gradsW[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.NUM_LAYERS):
            z = zs[-l]
            sp = self.sigmoidPrime(z)
            delta = np.dot(self.W[-l+1].transpose(), delta) * sp
            gradsB[-l] = delta
            gradsW[-l] = np.dot(delta, activations[-l-1].transpose())

        return (gradsB, gradsW)

    def gradientDescent(self, trainingSet, batchSize):
        trainingCost = []
        data = trainingSet.data

        for epoch in range(self.nEpochs):
            random.shuffle(data)
            self.cost = 0.0
            self.correct = 0
            batches = [data[k: k + batchSize] for k in range(0, trainingSet.size, batchSize)]

            for batch in batches:
                self.updateMiniBatch(batch, trainingSet.size)

            avgCost = self.cost / trainingSet.size
            trainingCost.append(avgCost)
            print('> epoch=%d, error=%.3f' % (epoch, avgCost))
            print('< correct=%d/%d' % (self.correct, trainingSet.size))
