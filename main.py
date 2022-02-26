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
# todo: add in testing set
# todo: implement ACCURACY_DIFF cutoff

import numpy as np
import pandas as pd

# "Training: Train perceptrons with three different learning rates:
# "   η = 0.001, 0.01, and 0.1.
ETA = (0.001, 0.01, 0.1)

# "Keep repeating until the accuracy on the training data has essentially stopped improving (i.e., the
# "difference between training accuracy from one epoch to the next is less than some small number,
# "like .01,) or you have run for 70 epochs (iterations through the training set), whichever comes first.
MAX_EPOCHS = 70
ACCURACY_DIFF = 0.01


# class for loading and preprocessing MNIST data
# data is contained in a numpy array
class Data:
    def __init__(self):
        self.TRAIN = "data/mnist_train.csv"
        self.TEST = "data/mnist_test.csv"
        self.training_data = self.load_set(self.TRAIN)
        self.testing_data = self.load_set(self.TEST)

    def load_set(self, dataset):
        print("Reading '" + dataset + "' data set...")
        data = pd.read_csv(dataset).to_numpy(dtype="float")
        return self.preprocess(data)

    # "Preprocessing: Scale each data value to be between 0 and 1.
    # "(i.e., divide each value by 255, which is the maximum value in the original data)
    # "This will help keep the weights from getting too large.
    @staticmethod
    def preprocess(data):
        max_value = 255
        print("Preprocessing data...")
        # for image_data in data:
        #    num_pixels = len(image_data) - 1
        #    for i in range(1, num_pixels + 1):
        #        image_data[i] /= max_value
        for image_data in data:
            temp = image_data[0]
            image_data /= 255
            image_data[0] = temp
        return data
        # wow I do see the loop is very slow

    def test(self):
        return self.testing_data

    def train(self):
        return self.training_data


# "You will use a perceptron with 785 inputs (including bias input)
# "and 10 outputs
class Perceptron:
    def __init__(self, eta):
        print("Initializing perceptron...")
        # Choose small random initial weights, 𝑤! ∈ [−.05, .05]
        # self.weights = np.random.uniform(-0.05, 0.05, (10, 785))
        self.weights = np.array([np.random.uniform(-0.05, 0.05, 785) for _ in range(10)])
        self.outputs = np.zeros(10)
        self.eta = eta

    # "Compute the accuracy on the training and test sets for this initial set of weights,
    # "to include in your plot. (Call this “epoch 0”.)
    def compute_accuracy(self, data, freeze=False):
        num_correct = 0

        # for each item in the dataset
        for d in data:
            # for each of the ten perceptrons
            for i in range(10):
                # "Recall that the bias unit is always set to 1,
                # "and the bias weight is treated like any other weight.
                temp = d[0]
                d[0] = 1
                # "Compute 𝒘 ∙ 𝒙 (i) at each output unit.
                self.outputs[i] = np.dot(self.weights[i], d)
                d[0] = temp

            # "If this is the correct prediction, don’t change the weights and
            # "go on to the next training example.
            if d[0] == np.argmax(self.outputs):
                num_correct += 1

            # "Otherwise, update all weights in the perceptron:
            # "    𝑤i ⟵ 𝑤i + 𝜂( 𝑡(i) − 𝑦(i) ) 𝑥i(i) , where
            # "
            # "    t(i) = { 1 if the output unit is the correct one for this training example
            # "           { 0 otherwise
            # "
            # "    y(i) = { 1 if 𝒘 ∙ 𝒙(i) > 0
            # "           { 0 otherwise
            # "
            # "Thus, 𝑡(i) − 𝑦(i) can be 1, 0, or −1.
            # "
            # "(Note that this means that for some output units 𝑡(i) − 𝑦(i) could be zero,
            # " and thus the weights to that output unit would not be updated,
            # " even if the prediction was incorrect. That’s okay!)
            elif not freeze:
                # for each perceptron
                for i in range(10):
                    ti = 1 if i == d[0] else 0
                    yi = 1 if self.outputs[i] > 0 else 0  # self.outputs[i] is already w dot x(i)
                    # np.add(ETA*(ti - yi), self.weights) # self.weights, out=self.weights,

                    # update the weights as a function of ti, yi, and the elements in both arrays
                    temp = d[0]
                    d[0] = 1
                    self.weights[i] = np.array([(wi + self.eta*(ti - yi)*xii) for wi, xii in zip(self.weights[i], d)])
                    d[0] = temp

        # return accuracy
        return num_correct / len(data.data)


def main():
    d = Data()

    p = Perceptron(ETA[1])
    print("Epoch 0: ", end="")
    print("Training Set:\tAccuracy:", p.compute_accuracy(d.train(), True), end="\t")
    print("Testing Set:\tAccuracy:", p.compute_accuracy(d.test(), True))

    for i in range(MAX_EPOCHS):
        print("Epoch " + str(i + 1) + ": ", end="")
        print("Training Set:\tAccuracy:", p.compute_accuracy(d.train()), end="\t")
        print("Testing Set:\tAccuracy:", p.compute_accuracy(d.test(), True))


if __name__ == '__main__':
    main()
