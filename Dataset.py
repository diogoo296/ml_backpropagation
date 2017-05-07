#!/usr/bin/env python
from csv import reader


class Dataset:
    MAX_VALUE = 255.0

    def __init__(self, filename):
        self.data = list()
        self.load(filename)

    def load(self, filename):
        with open(filename, 'r') as file:
            csvReader = reader(file)
            for row in csvReader:
                if not row:
                    continue
                self.data.append(self.formatAndNormalize(row))

    def formatAndNormalize(self, row):
        return [int(value.strip()) / self.MAX_VALUE for value in row]
