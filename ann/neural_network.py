import numpy as np
from .neural_layer import NeuralLayer

class NeuralNetwork:
    def __init__(self, config):
        self.num_hidden = config.number_hidden_layer
        self.num_neurons = config.number_neurons
        self.act_hidden = config.active_function_hidden
        self.act_output = config.active_function_output
        self.weight_ini_method = config.weight_ini_method
        self.lr = config.learning_rate

        # Build network architecture
        input_size = 784
        output_size = 10

        network_size = [input_size] + \
                       [self.num_neurons] * self.num_hidden + \
                       [output_size]

        self.layers = []

        for i in range(len(network_size) - 1):
            act = self.act_hidden if i < self.num_hidden else self.act_output
            layer = NeuralLayer(network_size[i], network_size[i+1], act, self.weight_ini_method)
            self.layers.append(layer)

    def forward(self, X):
        h = X
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def backward(self, y_true, y_pred):
        # Gradient of cross-entropy + softmax
        grad_a = (y_pred - y_true) / y_true.shape[0]

        # Backprop through layers (reverse order)
        for layer in reversed(self.layers):
            grad_a = layer.backward(grad_a)


    def update_weights(self):
        for layer in self.layers:
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append({
                "W": layer.W,
                "b": layer.b
            })
        return weights


    def set_weights(self, weights):
        for layer, w in zip(self.layers, weights):
            layer.W = w["W"]
            layer.b = w["b"]
