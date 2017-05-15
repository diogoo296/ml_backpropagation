#!/usr/bin/env python
import numpy as np
from theano import shared, grad, tensor as T


class NeuralNetwork:
    NUM_INPUTS = 784
    NUM_OUTPUTS = 10

    def __init__(self, numHiddenLayers, alpha, nEpochs):
        W_shape = (self.NUM_OUTPUTS, self.NUM_INPUTS)
        b_shape = self.NUM_OUTPUTS

        self.W = shared(np.random.random(W_shape) - 0.5, name="W")
        self.b = shared(np.random.random(b_shape) - 0.5, name="b")

        self.alpha = alpha
        self.nEpochs = nEpochs

nn = NeuralNetwork(1, 0.4, 1)

x = T.dmatrix("x")  # N x 784
labels = T.dmatrix("labels")  # N x 10
output = T.nnet.softmax(x.dot(nn.W.transpose()) + nn.b)
cost = T.nnet.binary_crossentropy(output, labels).mean()

grad_b = grad(cost, nn.b)
