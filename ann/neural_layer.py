import numpy as np
from .activations import activation, grad_der

class NeuralLayer:
    def __init__(self, in_size, out_size, activation_name, weight_ini_method):
        self.in_size = in_size
        self.out_size = out_size
        self.activation_name = activation_name

        # Weight initialization
        if weight_ini_method.lower() == "xavier":
            limit = np.sqrt(6 / (in_size + out_size))
            self.W = np.random.uniform(-limit, limit, (in_size, out_size))
        else:
            self.W = np.random.randn(in_size, out_size) * 0.01

        self.b = np.zeros((1, out_size))

        self.a = None
        self.h = None
        self.grad_W = None
        self.grad_b = None

    def forward(self, h_prev):
        self.x = h_prev
        self.a = np.dot(h_prev, self.W) + self.b
        self.h = activation(self.a, self.activation_name)
        return self.h

    def backward(self, grad_h):
        # If not output layer, apply activation derivative
        if self.activation_name != "softmax":
            grad_h *= grad_der(self.a, self.activation_name)

        self.grad_W = np.dot(self.x.T, grad_h)
        self.grad_b = np.sum(grad_h, axis=0, keepdims=True)

        grad_h_prev = np.dot(grad_h, self.W.T)
        return grad_h_prev

