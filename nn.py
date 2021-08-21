import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        # layer_sizes example: [4, 10, 2]

        self.input_layer_size, self.hidden_layer_size, self.output_layer_size = layer_sizes

        # WEIGHTS
        self.W1 = np.random.normal(size=(self.hidden_layer_size, self.input_layer_size))
        self.W2 = np.random.normal(size=(self.output_layer_size, self.hidden_layer_size))

        # BIASES
        self.b1 = np.zeros((self.hidden_layer_size, 1))
        self.b2 = np.zeros((self.output_layer_size, 1))

    def activation(self, x):
        # SIGMOID
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        # x example: np.array([[0.1], [0.2], [0.3]])

        a1 = self.activation(self.W1 @ x + self.b1)
        a2 = self.activation(self.W2 @ a1 + self.b2)

        return a2
