import numpy as np

def sigmoid(x):
  return 1 /(1 + np.exp(-x))
def relu(x):
  return np.maximum(0, x)
def tanh(x):
	return np.tanh(x)
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def activation(x,active_function):
  if active_function=="sigmoid":
    return sigmoid(x)
  if active_function=="relu":
    return relu(x)
  if active_function=="tanh":
    return tanh(x)
  if active_function=="softmax":
    return softmax(x)

def der_sigmoid(x):
  return sigmoid(x)*(1-sigmoid(x))
def der_relu(x):
  return np.where(x<=0,0,1)
def der_tanh(x):
  return 1-(tanh(x)**2)
def softmax_derivative(softmax_output):
    s = softmax_output.reshape(-1, 1)
    jacobian = np.diagflat(s) - np.dot(s, s.T)
    return jacobian
def grad_der(x, active_fun):
    if active_fun == "relu":
        return (x > 0).astype(float)
    elif active_fun == "sigmoid":
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)
    elif active_fun == "tanh":
        return 1 - np.tanh(x) ** 2
    else:
        return 1  # for softmax (handled in loss gradient)

