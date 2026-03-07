import wandb
from train import train
from utils.data_loader import load_data

# Load data once globally
X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()

sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [5, 10]},
        "number_hidden_layer": {"values": [3, 4, 5]},
        "number_neurons": {"values": [32, 64, 128]},
        "learning_rate": {"values": [1e-3, 1e-4]},
        "optimizer": {"values": ["sgd", "momentum", "nag", "rmsprop"]},
        "batch_size": {"values": [16, 32, 64]},
        "weight_ini_method": {"values": ["random", "xavier"]},
        "active_function_hidden": {"values": ["sigmoid", "tanh"]},
        "active_function_output": {"values": ["softmax"]},
        "loss": {"values": ["cross_entropy"]},
        "epsilon": {"values": [1e-8]},
        "beta": {"values": [0.9]},
        "beta1": {"values": [0.9]},
        "beta2": {"values": [0.99]},
        "weight_decay": {"values": [0, 0.0005, 0.5]}
    }
}

# Create sweep in ASSIGNMENT_1 project
sweep_id = wandb.sweep(
    sweep=sweep_config,
    project="ASSIGNMENT_1"
)


def main():

    with wandb.init(project="ASSIGNMENT_1"):

        config = wandb.config

        run_name = (
            "act-" + config.active_function_hidden +
            "_hl-" + str(config.number_hidden_layer) +
            "_ep-" + str(config.epochs) +
            "_neur-" + str(config.number_neurons) +
            "_wd-" + str(config.weight_decay) +
            "_lr-" + str(config.learning_rate) +
            "_opt-" + config.optimizer +
            "_bs-" + str(config.batch_size)
        )

        wandb.run.name = run_name

        train(config)


wandb.agent(sweep_id, function=main, count=100)

wandb.finish()
