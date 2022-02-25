# CS445 Homework 1 Problem 11
# Christopher Juncker
# This is a sample Python script.

# "11. This is a short coding problem. You will use a perceptron with 785 inputs (including bias input)
# "and 10 outputs to learn to classify the handwritten digits in the MNIST dataset
# "(http://yann.lecun.com/exdb/mnist/). See the class slides for details of the perceptron architecture
# "and perceptron learning algorithm.
#
# MNIST data in CSV format:
# https://pjreddie.com/projects/mnist-in-csv/
#
#

import numpy as np
import pandas as pd


# LOADING MNIST DATA

class Data:
    def __init__(self):
        self.TRAIN = "data/mnist_train.csv"
        self.TEST = "data/mnist_test.csv"
        self.data = None

    def load_training_set(self):
        self.load_set(self.TRAIN)

    def load_test_set(self):
        self.load_set(self.TEST)

    def load_set(self, dataset):
        self.data = pd.read_csv(dataset).to_numpy(dtype="float")
        self.preprocess()

    # Preprocessing: Scale each data value to be between 0 and 1.
    # (i.e., divide each value by 255, which is the maximum value in the original data)
    # This will help keep the weights from getting too large.
    def preprocess(self):
        max_value = 255

        for image_data in self.data:
            num_pixels = len(image_data) - 1
            for i in range(1, num_pixels + 1):
                image_data[i] /= max_value


def main():
    data = Data()
    data.load_test_set()

    for d in data.data:
        print(d)


if __name__ == '__main__':
    main()
