import numpy as np

def cross_entropy(y_pred, y_true):
   epsilon=1e-15
   y_pred=np.clip(y_pred,epsilon,1-epsilon)
   loss=-np.mean(np.sum(y_true*np.log(y_pred),axis=1))
   return loss

def mse_loss(y_pred, y_true):
    loss = np.mean(np.sum((y_true - y_pred) ** 2, axis=1))
    return loss

