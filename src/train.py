import argparse
import numpy as np
import wandb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import confusion_matrix, f1_score

from src.utils.data_loader import load_data
from src.ann.neural_network import NeuralNetwork
from src.ann.objective_functions import mse_loss
from src.ann.optimizers import sgd, momentum, nag, rmsprop


# -----------------------------
# Loss Function
# -----------------------------
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))


# -----------------------------
# Log Sample Images to W&B
# -----------------------------
def log_sample_images(X_train, y_train):
    table = wandb.Table(columns=["Digit", "Image"])
    samples_per_class = 5
    class_counts = {i: 0 for i in range(10)}

    labels = np.argmax(y_train, axis=1)

    for img, label in zip(X_train, labels):
        if class_counts[label] < samples_per_class:
            table.add_data(
                str(label),
                wandb.Image(img.reshape(28, 28))
            )
            class_counts[label] += 1

        if all(count == samples_per_class for count in class_counts.values()):
            break

    wandb.log({"MNIST Sample Images": table})


# -----------------------------
# Training Function
# -----------------------------
def train(config):

    print("Starting training with config:")
    for k, v in vars(config).items():
        print(f"{k}: {v}")

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    print("Train:", X_train.shape, y_train.shape)
    print("Val:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)

    log_sample_images(X_train, y_train)

    # Initialize Neural Network
    nn = NeuralNetwork(config)

    # Optimizer states
    t = 0
    velocity = None
    cache = None
    m = None
    v = None
    best_f1 = 0
    best_weights = None
    best_config = None

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(config.epochs):

        t += 1

        # -------- TRAIN --------
        logits = nn.forward(X_train)

        # Monitor activations of first hidden layer
        first_layer_output = nn.layers[0].a   # activation output
        zero_fraction = np.mean(first_layer_output == 0)

        wandb.log({
            "dead_neuron_fraction_layer1": zero_fraction
        })

        wandb.log({
            "activations_layer1": wandb.Histogram(first_layer_output)
        })

        # Compute training accuracy
        train_predictions = np.argmax(logits, axis=1)
        train_true = np.argmax(y_train, axis=1)
        train_accuracy = np.mean(train_predictions == train_true)
        
        if config.loss == "cross_entropy":
            loss = cross_entropy_loss(y_train, logits)
        elif config.loss == "mse":
            loss = mse_loss(logits, y_train)
        else:
            raise ValueError("Unsupported loss function")

        nn.backward(y_train, logits)

        # Compute gradient norm of first hidden layer weights
        grad_norm = np.linalg.norm(nn.layers[0].grad_W)

        # -------- OPTIMIZER STEP --------
        if config.optimizer == "sgd":
            sgd(nn, config.learning_rate, config.weight_decay)

        elif config.optimizer == "momentum":
            velocity = momentum(
                nn, config.learning_rate, config.beta,
                velocity, config.weight_decay
            )

        elif config.optimizer == "nag":
            velocity = nag(
                nn, config.learning_rate, config.beta,
                velocity, config.weight_decay
            )

        elif config.optimizer == "rmsprop":
            cache = rmsprop(
                nn, config.learning_rate, config.beta,
                config.epsilon, cache, config.weight_decay
            )

        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")

        # -------- VALIDATION (After Backward!) --------
        val_logits = nn.forward(X_val)
        val_predictions = np.argmax(val_logits, axis=1)
        val_true = np.argmax(y_val, axis=1)
        val_accuracy = np.mean(val_predictions == val_true)

        # -------- TEST --------
        test_logits = nn.forward(X_test)
        test_predictions = np.argmax(test_logits, axis=1)
        test_true = np.argmax(y_test, axis=1)
        test_accuracy = np.mean(test_predictions == test_true)

        test_f1 = f1_score(test_true, test_predictions, average="macro")

        print(
            f"Epoch {epoch+1}/{config.epochs}, "
            f"Loss: {loss:.4f}, "
            f"Val Acc: {val_accuracy:.4f}"
            f"Test F1: {test_f1:.4f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "loss": loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "grad_norm_layer1": grad_norm,
            "test_f1": test_f1
        })

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_weights = nn.get_weights()
            best_config = dict(config)

    np.save("best_model.npy", best_weights)

    with open("best_config.json", "w") as f:
        json.dump(best_config, f, indent=4)

    print("Best model saved as best_model.npy")
    print("Best configuration saved as best_config.json")
    

    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------
    cm = confusion_matrix(test_true, test_predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Test Set")
    plt.savefig("confusion_matrix.png")

    wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})

    # Save model
    with open("model.pkl", "wb") as f:
        pickle.dump(nn, f)

    print("Training finished! Model saved as model.pkl")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="rmsprop")
    parser.add_argument("--number_hidden_layer", type=int, default=2)
    parser.add_argument("--number_neurons", type=int, default=128)
    parser.add_argument("--active_function_hidden", type=str, default="sigmoid")
    parser.add_argument("--active_function_output", type=str, default="softmax")
    parser.add_argument("--weight_ini_method", type=str, default="xavier")
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--loss", type=str, default="cross_entropy")

    args = parser.parse_args()

    wandb.init(project="ASSIGNMENT_1", config=vars(args))
    config = wandb.config

    train(config)


if __name__ == "__main__":
    main()
