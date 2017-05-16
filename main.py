#!/usr/bin/env python
from dataset import Dataset
from neural_network import NeuralNetwork


dataset = Dataset('data.csv')
nn = NeuralNetwork(50, 1, 100)
nn.sgd(dataset)
