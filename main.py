# CS445 Homework 1 Problem 11
# Christopher Juncker
# This is a sample Python script.

# "11. This is a short coding problem. You will use a perceptron with 785 inputs (including bias input)
# "and 10 outputs to learn to classify the handwritten digits in the MNIST dataset
# "(http://yann.lecun.com/exdb/mnist/). See the class slides for details of the perceptron architecture
# "and perceptron learning algorithm.
#
#
# MNIST data in CSV format:
# https://pjreddie.com/projects/mnist-in-csv/
#
# Data files in data/ folder:
#   mnist_train.csv
#   mnist_test.csv
# (Not included in commit to save space)
#
#

import numpy as np
import pandas as pd


# class for loading and preprocessing MNIST data
# data is contained in a numpy array
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
        print("Reading '" + dataset + "' data set...")
        self.data = pd.read_csv(dataset).to_numpy(dtype="float")
        self.preprocess()

    # "Preprocessing: Scale each data value to be between 0 and 1.
    # "(i.e., divide each value by 255, which is the maximum value in the original data)
    # "This will help keep the weights from getting too large.
    def preprocess(self):
        max_value = 255
        print("Preprocessing data...")
        for image_data in self.data:
            num_pixels = len(image_data) - 1
            for i in range(1, num_pixels + 1):
                image_data[i] /= max_value


# "You will use a perceptron with 785 inputs (including bias input)
# "and 10 outputs
class Perceptron:
    def __init__(self):
        print("Initializing perceptron...")
        # Choose small random initial weights, 𝑤! ∈ [−.05, .05]
        self.weights = np.random.uniform(-0.05, 0.05, (10, 785))
        self.outputs = np.zeros(10)
        self.epoch = 0

    # "Compute the accuracy on the training and test sets for this initial set of weights,
    # "to include in your plot. (Call this “epoch 0”.)
    def compute_accuracy(self, data):
        print("Epoch " + str(self.epoch) + ": ", end="")
        self.epoch += 1
        num_correct = 0

        # for each item in the dataset
        for d in data.data:
            # for each of the ten perceptrons
            for i in range(10):
                # "Recall that the bias unit is always set to 1,
                # "and the bias weight is treated like any other weight.
                temp = d[0]
                d[0] = 1
                self.outputs[i] = np.dot(self.weights[i], d)
                d[0] = temp

            if d[0] == np.argmax(self.outputs):
                num_correct += 1

        print("\tAccuracy:", num_correct / len(data.data))











def main():
    d = Data()
    d.load_test_set()

    p = Perceptron()
    p.compute_accuracy(d)






if __name__ == '__main__':
    main()
