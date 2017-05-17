#!/usr/bin/env python
from dataset import Dataset
from neural_network import NeuralNetwork


dataset = Dataset('data2k.csv')
nn = NeuralNetwork(25, 0.5, 100)
nn.sgd(dataset)
