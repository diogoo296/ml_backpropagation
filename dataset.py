#!/usr/bin/env python
from csv import reader
import numpy as np


class Dataset:
    MAX_VALUE = 255.0

    def __init__(self, filename):
        self.data = list()
        self.load(filename)
        self.size = len(self.data)

    def load(self, filename):
        with open(filename, 'r') as file:
            csvReader = reader(file)
            for row in csvReader:
                if not row:
                    continue
                formatedRow = self.formatAndNormalize(row[1:len(row)])
                formatedRow.append(int(row[0].strip()))
                self.data.append(np.array(formatedRow))

    def formatAndNormalize(self, row):
        return [int(value.strip()) / self.MAX_VALUE for value in row]
