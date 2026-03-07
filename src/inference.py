import argparse
import numpy as np
import json

from sklearn.metrics import f1_score, precision_score, recall_score

from src.utils.data_loader import load_data
from src.ann.neural_network import NeuralNetwork
from src.ann.objective_functions import cross_entropy


def load_model(model_path):
    data = np.load(model_path, allow_pickle=True)
    return data


def evaluate_model(model, X_test, y_test):

    logits = model.forward(X_test)

    loss = cross_entropy(logits, y_test)

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(logits, axis=1)

    accuracy = np.mean(y_true == y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="best_model.npy")
    parser.add_argument("--config_path", type=str, default="best_config.json")

    args = parser.parse_args()

    print("Loading configuration...")
    config = json.load(open(args.config_path))

    class CLI:
        pass

    cli = CLI()
    for k, v in config.items():
        setattr(cli, k, v)

    print("Building model...")
    model = NeuralNetwork(cli)

    print("Loading weights...")
    weights = load_model(args.model_path)
    model.set_weights(weights)

    print("Loading test data...")
    _, _, _, _, X_test, y_test = load_data()

    print("Running inference...")
    results = evaluate_model(model, X_test, y_test)

    print("\nResults:")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")


if __name__ == "__main__":
    main()
