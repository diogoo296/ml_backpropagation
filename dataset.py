#!/usr/bin/env python
from csv import reader


class Dataset:
    MAX_VALUE = 255.0

    def __init__(self, filename):
        self.inputs = list()
        self.outputs = list()
        self.load(filename)

    def load(self, filename):
        with open(filename, 'r') as file:
            csvReader = reader(file)
            for row in csvReader:
                if not row:
                    continue
                self.inputs.append(self.formatAndNormalize(row[1:len(row)]))
                self.outputs.append(int(row[0].strip()))

    def formatAndNormalize(self, row):
        return [int(value.strip()) / self.MAX_VALUE for value in row]
