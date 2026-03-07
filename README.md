# Deep_Learning_assignment_1


# Overview

This assignment focuses on implementing a configurable and modular Multi-Layer Perceptron (MLP) using NumPy. The objective is to understand the internal working of neural networks by building the complete training pipeline from scratch. The key components includes forward propagation, backpropagation, loss computation, and multiple optimization strategies. The model is designed to be flexible, allowing different architectures, activation functions, and hyperparameters to be configured easily. The developed MLP is trained and evaluated on the MNIST and Fashion-MNIST datasets to perform image classification tasks.


[Wandb Report Link](https://wandb.ai/ma24m006-indian-institute-of-technology-madras/ASSIGNMENT_1/reports/-DA6401_Assignment_1--VmlldzoxNjExNzI4OA?accessToken=4gq97o3kmtw99i0r43c40d5oe2vv1lt0xpksovqrr2q68threualbyrsbkv1v1ye)


[Github Repo Link](https://github.com/BritiJadav/da6401_assignment_1)


The skeleton of the assignment looks like : 

```
|--- src/
       |---ann/
                    |-- __init__.py             
                    │-- activations.py                     
                    |-- neural_layer.py                    
                    |-- neural_network.py
                    |-- objective_functions.py
                    |-- optimizers.py 
       |---utils/
                    |-- __init__.py
                    |-- data_loader.py
       |--- inference.py
       |--- train.py
       |--- best_config.json
       --- best_model.npy

```
