#!/usr/bin/env python
from dataset import Dataset
from neural_network2 import NeuralNetwork


dataset = Dataset('data.csv')
nn = NeuralNetwork(25, 1, 1000)
nn.gradientDescent(dataset, 1)
