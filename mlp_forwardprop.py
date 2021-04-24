import numpy as np
from random import random

# save the activations and derivatives
# implement back propagation
# implement gradient descent
# implement train
# train our network with some dummy dataset
# make predictions


class MLP:

    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        # initiate random weights
        self.weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i+1])  # create a random matrix with dimensions layers[i] * layers[i+1]
            self.weights.append(w)

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives

    def forward_propagate(self, inputs):
        activations = inputs
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
            # calculate the net inputs
            net_inputs = np.dot(activations, w)
            # calculate the activations
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations

        return activations

    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def back_propagate(self, error, verbose=False):
        # dE/dW_i = (y-a_[i+1]) s`(h_[i+1]) a_i
        # s`(h_[i+1]) = s(h_[i+1]) (1-s(h_[i+1]))
        # s(h_[i+1]) = a_[i+1]
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))
        return error

    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):

        for i in range(epochs):
            sum_error = 0
            for input, target in zip(inputs, targets):
                # forward propagation
                output = self.forward_propagate(input)

                # calculate the error
                error = target - output

                # back propagation
                self.back_propagate(error)

                # apply gradient descent
                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output)

            # report error
            print("Error: {} at epoch {}".format(sum_error/len(inputs), i))

    def _mse(self, target, output):
        return np.average((target - output) ** 2)

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)


if __name__ == '__main__':

    # create a dataset to train a network for the sum operation
    inputs = np.array([[random() / 2 for i in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])

    # create an mlp
    mlp = MLP(2, [5], 1)

    # train our mlp
    mlp.train(inputs, targets, 150, 0.2)

    # create dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    output = mlp.forward_propagate(input)
    print()
    print()
    print()
    print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output))
