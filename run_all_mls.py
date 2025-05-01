
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import time
import json
import pickle

from sklearn.model_selection import train_test_split

import utils
from data_loader import load_test_data, load_train_data
from main_train_ml import train_ml_model, hyperparameter_search, save_model

os.environ["MLIR_CRASH_REPRODUCER_DIRECTORY"] = "0"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run all anomaly detection models")
    parser.add_argument("--dataset", type=str, default="BATADAL", help="Dataset to use")
    parser.add_argument(
        "--run_name", type=str, default="comparison", help="Name of this run"
    )
    parser.add_argument("--gpus", type=str, default="0", help="GPUs to use")
    parser.add_argument(
        "--test_split", type=float, default=0.7, help="Test split ratio"
    )
    parser.add_argument("--history", type=int, default=100, help="History window size")
    parser.add_argument(
        "--model_types",
        type=str,
        default=None,
        help="Comma-separated list of model types to run (e.g., 'RF,SVM,ADA'). If not provided, all models will run.",
    )
    return parser.parse_args()


def setup_model_config(model_type, history):
    config = {
        "name": f"{model_type}-model",
        "model": {"history": history},
        "grid_search": {
            "percentile": [0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999],
            "window": [1, 5, 10, 20, 30, 50, 100],
            "metrics": ["F1"],
            "detection-plots": True,
            "save-metric-info": True,
            "save-theta": True,
        },
    }

    if model_type == "SVM":
        config["model"]["C"] = 1.0
        config["model"]["kernel"] = "rbf"
        config["model"]["epsilon"] = 0.1
        config["model"]["gamma"] = "scale"
        config["model"]["random_state"] = 42
        config["model"]["verbose"] = 1
    elif model_type == "KNN":
        config["model"]["n_neighbors"] = 5
        config["model"]["weights"] = "uniform"
        config["model"]["algorithm"] = "auto"
        config["model"]["p"] = 2
        config["model"]["verbose"] = 1
    elif model_type == "MLP":
        config["model"]["hidden_layer_sizes"] = (100, 50)
        config["model"]["activation"] = "relu"
        config["model"]["solver"] = "adam"
        config["model"]["alpha"] = 0.0001
        config["model"]["learning_rate"] = "adaptive"
        config["model"]["max_iter"] = 200
        config["model"]["random_state"] = 42
        config["model"]["verbose"] = 1
    elif model_type == "ADA":
        config["model"]["n_estimators"] = 100
        config["model"]["learning_rate"] = 1.0
        config["model"]["loss"] = "linear"
        config["model"]["random_state"] = 42
        config["model"]["verbose"] = 1

    return config


def evaluate_model(
    model_type, config, Xfull, Xtest, Ytest, dataset_name, run_name, test_split
):
    start_time = time.time()

    print(f"\n=============================================")
    print(f"Training and evaluating {model_type} model...")
    print(f"=============================================\n")
    Xtrain, Xval, _, _ = train_test_split(
        Xfull, Xfull, test_size=0.2, random_state=42, shuffle=False
    )

    event_detector = train_ml_model(model_type, config, Xtrain, Xval)

    hyperparameter_search(
        event_detector,
        model_type,
        config,
        Xval,
        Xtest,
        Ytest,
        dataset_name,
        test_split=test_split,
        run_name=run_name,
        verbose=1,
    )
    save_model(event_detector, config, run_name)

    end_time = time.time()
    training_time = end_time - start_time

    print(
        f"Training and evaluation of {model_type} completed in {training_time:.2f} seconds\n"
    )

    return event_detector, training_time


def evaluate_all_models(args):
    # Load data
    print("Loading data...")
    Xfull, sensor_cols = load_train_data(args.dataset)
    Xtest, Ytest, _ = load_test_data(args.dataset)
    print(
        f"Data loaded: Xfull shape: {Xfull.shape}, Xtest shape: {Xtest.shape}, Ytest shape: {Ytest.shape}"
    )

    # Create all output directories if they don't exist
    os.makedirs(f"models/{args.run_name}", exist_ok=True)
    os.makedirs(f"plots/{args.run_name}", exist_ok=True)
    os.makedirs(f"results/{args.run_name}", exist_ok=True)
    os.makedirs(f"npys/{args.run_name}", exist_ok=True)
    os.makedirs("npys/results", exist_ok=True)

    all_model_types = ["SVM", "MLP", "ADA", "KNN"]

    if args.model_types:
        model_types = [model.strip() for model in args.model_types.split(",")]
        print(f"Running selected models: {', '.join(model_types)}")
    else:
        model_types = all_model_types
        print(f"Running all models: {', '.join(model_types)}")
    results = {
        "model_type": [],
        "training_time": [],
        "f1_micro": [],
        "f1_macro": [],
        "precision_micro": [],
        "precision_macro": [],
        "recall_micro": [],
        "recall_macro": [],
        "accuracy": [],
    }

    # For ROC curve
    roc_curves = {}

    Xtest_val, Xtest_test, Ytest_val, Ytest_test = utils.custom_train_test_split(
        args.dataset, Xtest, Ytest, test_size=args.test_split, shuffle=False
    )

    for model_type in model_types:
        config = setup_model_config(model_type, args.history)

        # Train and evaluate model
        event_detector, training_time = evaluate_model(
            model_type,
            config,
            Xfull,
            Xtest,
            Ytest,
            args.dataset,
            args.run_name,
            args.test_split,
        )

        try:
            best_theta_file = f"models/{args.run_name}/{config['name']}-theta.pkl"
            if not os.path.exists(best_theta_file):
                best_theta_file = f"models/results/{config['name']}-theta.pkl"

            with open(best_theta_file, "rb") as f:
                best_theta = pickle.load(f)

            with open(f"models/{args.run_name}/{config['name']}.json", "r") as f:
                params = json.load(f)
                best_window = params.get("best_window", 1)
        except:
            best_theta = np.quantile(
                event_detector.reconstruction_errors(Xfull).mean(axis=1), 0.95
            )
            best_window = 1

        history = config["model"]["history"]
        if history > 0:
            final_test_data = Xtest_test
            final_test_labels = Ytest_test[history + 1 :]
        else:
            final_test_data = Xtest_test
            final_test_labels = Ytest_test

        # Get reconstruction errors
        final_test_errors = event_detector.reconstruction_errors(final_test_data)
        final_test_instance_errors = final_test_errors.mean(axis=1)

        final_Yhat = event_detector.cached_detect(
            final_test_instance_errors, best_theta, best_window
        )
        print("Final Yhat shape:", final_Yhat.shape)
        print("Final test labels shape:", final_test_labels.shape)
        final_test_labels = final_test_labels.reshape(-1)

        f1_micro = f1_score(final_test_labels, final_Yhat, average="micro")
        f1_macro = f1_score(final_test_labels, final_Yhat, average="macro")
        precision_micro = precision_score(
            final_test_labels, final_Yhat, average="micro"
        )
        precision_macro = precision_score(
            final_test_labels, final_Yhat, average="macro"
        )
        recall_micro = recall_score(final_test_labels, final_Yhat, average="micro")
        recall_macro = recall_score(final_test_labels, final_Yhat, average="macro")
        accuracy = (final_Yhat == final_test_labels).mean()

        results["model_type"].append(model_type)
        results["training_time"].append(training_time)
        results["f1_micro"].append(f1_micro)
        results["f1_macro"].append(f1_macro)
        results["precision_micro"].append(precision_micro)
        results["precision_macro"].append(precision_macro)
        results["recall_micro"].append(recall_micro)
        results["recall_macro"].append(recall_macro)
        results["accuracy"].append(accuracy)

        fpr, tpr, _ = roc_curve(final_test_labels, final_test_instance_errors)
        # save fpr, tpr, auc
        with open(f"results/{args.run_name}/{model_type}_roc.pkl", "wb") as f:
            pickle.dump((fpr, tpr, auc(fpr, tpr)), f)
        roc_curves[model_type] = (fpr, tpr, auc(fpr, tpr))
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results/{args.run_name}/model_comparison.csv", index=False)
    print("\nModel Performance Comparison:")
    print(results_df.to_string())

    plt.figure(figsize=(10, 8))

    colors = ["blue", "red", "green", "orange", "purple"]
    line_styles = ["-", "--", ":", "-.", "-"]

    for i, model_type in enumerate(model_types):
        if model_type in roc_curves:
            fpr, tpr, roc_auc = roc_curves[model_type]
            plt.plot(
                fpr,
                tpr,
                color=colors[i % len(colors)],
                linestyle=line_styles[i % len(line_styles)],
                lw=2,
                label=f"{model_type} (AUC = {roc_auc:.3f})",
            )

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2, label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Anomaly Detection Models")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.savefig(f"plots/{args.run_name}/roc_comparison.pdf")
    # plt.savefig(f"plots/{args.run_name}/roc_comparison.png")
    # print(f"ROC curves saved to plots/{args.run_name}/roc_comparison.pdf and .png")
    best_model_idx = results_df["f1_macro"].idxmax()
    best_model = results_df.iloc[best_model_idx]
    print(
        f"\nBest model by F1 (macro): {best_model['model_type']} with F1 = {best_model['f1_macro']:.4f}"
    )

    return results_df


if __name__ == "__main__":
    args = parse_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    evaluate_all_models(args)
