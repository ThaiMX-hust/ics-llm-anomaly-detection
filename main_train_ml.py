"""

Copyright 2020 Lujo Bauer, Clement Fung

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import argparse
import json
import os
import pdb
import pickle
import sys
import time
from sklearn.multioutput import MultiOutputRegressor

import warnings
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

import pandas as pd
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    precision_score,
    recall_score,
    classification_report,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import metrics
import utils
from data_loader import load_test_data, load_train_data

from detector import (
    autoencoder,
    cnn,
    dnn,
    gru,
    identity,
    linear,
    lstm,
    svm,
    knn,
    mlp,
    adaboost,
)

os.environ["MLIR_CRASH_REPRODUCER_DIRECTORY"] = "0"
print(tf.config.list_physical_devices("GPU"))


def train_reconstruction_model(model_type, config, Xtrain, Xval):

    train_params = config["train"]
    model_params = config["model"]

    # define model input size parameter --- needed for AE size
    model_params["nI"] = Xtrain.shape[1]

    if model_type == "AE":
        event_detector = autoencoder.AEED(**model_params)
    elif model_type == "ID":
        event_detector = identity.Identity(**model_params)
    else:
        print(f"Model type {model_type} is not supported.")
        return

    event_detector.create_model()

    event_detector.train(Xtrain, validation_data=(Xval, Xval), **train_params)

    return event_detector


def train_forecast_model(model_type, config, Xtrain, Xval, Ytrain, Yval):

    train_params = config["train"]
    model_params = config["model"]

    # define model input size parameter --- needed for AE size
    model_params["nI"] = Xtrain.shape[2]

    if model_type == "GRU":
        event_detector = gru.GatedRecurrentUnit(**model_params)
    elif model_type == "LSTM":
        event_detector = lstm.LongShortTermMemory(**model_params)
    elif model_type == "DNN":
        event_detector = dnn.DeepNN(**model_params)
    elif model_type == "CNN":
        event_detector = cnn.ConvNN(**model_params)
    elif model_type == "LIN":
        event_detector = linear.Linear(**model_params)
    elif model_type == "ID":
        event_detector = identity.Identity(**model_params)
    elif model_type == "RF":
        event_detector = rf.RFRegressor(**model_params)
    else:
        print(f"Model type {model_type} is not supported.")
        return

    event_detector.create_model()

    event_detector.train(Xtrain, Ytrain, validation_data=(Xval, Yval), **train_params)

    return event_detector


def train_forecast_model_by_idxs(model_type, config, Xfull, train_idxs, val_idxs):

    train_params = config["train"]
    model_params = config["model"]

    # define model input size parameter --- needed for AE size
    model_params["nI"] = Xfull.shape[1]

    if model_type == "GRU":
        event_detector = gru.GatedRecurrentUnit(**model_params)
    elif model_type == "LSTM":
        event_detector = lstm.LongShortTermMemory(**model_params)
    elif model_type == "DNN":
        event_detector = dnn.DeepNN(**model_params)
    elif model_type == "CNN":
        event_detector = cnn.ConvNN(**model_params)
    elif model_type == "LIN":
        event_detector = linear.Linear(**model_params)
    elif model_type == "ID":
        event_detector = identity.Identity(**model_params)
    elif model_type == "RF":
        event_detector = rf.RFRegressor(**model_params)
    else:
        print(f"Model type {model_type} is not supported.")
        return

    event_detector.create_model()

    event_detector.train_by_idx(
        Xfull, train_idxs, val_idxs, validation_data=True, **train_params
    )
    return event_detector


def train_ml_model(model_type, config, Xtrain, Ytrain):
    """
    General function to train various Machine Learning algorithms (RF, XGBoost, SVM, KNN, MLP, etc.).

    Args:
        model_type (str): Type of model to train (RF, XG, SVM, KNN, MLP, ADA).
        config (dict): Model configuration.
        Xtrain (np.array): Training input data.
        Ytrain (np.array): Training target data.

    Returns:
        event_detector: Trained ML model.
    """

    model_params = config["model"]

    if model_type == "RF":
        print("Initializing Random Forest Regressor...")
        event_detector = rf.RFRegressor(**model_params)
    elif model_type == "XG":
        print("Initializing XGBoost Regressor...")
        event_detector = Xgboost.XGBoostRegressor(**model_params)
    elif model_type == "SVM":
        print("Initializing SVM Regressor...")
        event_detector = svm.SVMRegressor(**model_params)
    elif model_type == "KNN":
        print("Initializing KNN Regressor...")
        event_detector = knn.KNNRegressor(**model_params)
    elif model_type == "MLP":
        print("Initializing MLP Regressor...")
        event_detector = mlp.MLPRegressorModel(**model_params)
    elif model_type == "ADA":
        print("Initializing AdaBoost Regressor...")
        event_detector = adaboost.AdaBoostRegressorModel(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if Ytrain.shape != Xtrain.shape:
        print(
            f"Note: Target shape {Ytrain.shape} doesn't match input shape {Xtrain.shape}. Using input as target for reconstruction."
        )
        Ytrain = Xtrain.copy()

    # Transform data into windowed format
    Xtrain_windowed, Ytrain_windowed = event_detector.transform_to_window_data(
        Xtrain, Ytrain
    )

    print("X_train_windowed shape:", Xtrain_windowed.shape)
    print("Y_train_windowed shape:", Ytrain_windowed.shape)

    if len(Ytrain_windowed.shape) > 1 and Ytrain_windowed.shape[1] > 1:
        pass

    event_detector.train(Xtrain_windowed, Ytrain_windowed)

    print(f"{model_type} Training Completed!")

    return event_detector


def hyperparameter_search(
    event_detector,
    model_type,
    config,
    Xval,
    Xtest,
    Ytest,
    dataset_name,
    val_idxs=None,
    test_split=0.7,
    run_name="results",
    verbose=1,
):
    start_time = time.time()

    print("Xval shape: ", Xval.shape)
    model_name = config["name"]
    do_batches = False

    Ytest = Ytest.astype(int)
    Xtest_val, Xtest_test, Ytest_val, Ytest_test = utils.custom_train_test_split(
        dataset_name, Xtest, Ytest, test_size=test_split, shuffle=False
    )

    if not model_type == "AE":

        history = event_detector.params["history"]

    history = 100
    # Clip the prediction to match LSTM prediction window
    Ytest_test = Ytest_test[history + 1 :]
    Ytest_val = Ytest_val[history + 1 :]
    do_batches = True

    ##### Cross Validation
    if val_idxs is None:
        validation_errors = event_detector.reconstruction_errors(
            Xval, batches=do_batches
        )
    else:
        validation_errors = utils.reconstruction_errors_by_idxs(
            event_detector, Xval, val_idxs, history
        )

    # MSE
    test_errors = event_detector.reconstruction_errors(Xtest_val, batches=do_batches)
    test_instance_errors = test_errors.mean(axis=1)
    print("MSE: ", test_instance_errors)

    grid_config = config.get("grid_search", dict())

    cutoffs = grid_config.get("percentile", [0.95])
    windows = grid_config.get("window", [1])
    eval_metrics = grid_config.get("metrics", ["F1"])

    firstPlotsError = True
    firstNpysError = True

    os.makedirs(f"plots/{run_name}", exist_ok=True)
    os.makedirs(f"plots/results", exist_ok=True)
    os.makedirs(f"npys/{run_name}", exist_ok=True)
    os.makedirs("npys/results", exist_ok=True)
    os.makedirs(f"models/{run_name}", exist_ok=True)
    os.makedirs("models/results", exist_ok=True)

    for metric in eval_metrics:

        negative_metric = metric == "false_positive_rate"

        if negative_metric:
            best_metric = 1
        else:
            best_metric = -1000

        best_percentile = 0
        best_window = 0
        metric_vals = np.zeros((len(cutoffs), len(windows)))
        metric_func = metrics.get(metric)

        for percentile_idx in range(len(cutoffs)):

            percentile = cutoffs[percentile_idx]

            # set threshold as quantile of average reconstruction error
            theta = np.quantile(validation_errors.mean(axis=1), percentile)

            for window_idx in range(len(windows)):

                window = windows[window_idx]

                # Yhat = event_detector.detect(Xtest, theta = theta, window = window, batches=True)
                Yhat = event_detector.cached_detect(
                    test_instance_errors, theta=theta, window=window
                )
                # Yhat = Yhat[window-1:].astype(int)
                print("Yhat", Yhat)
                print("Ytest_val", Ytest_val)
                Yhat, Ytest_val = utils.normalize_array_length(Yhat, Ytest)
                choice_value = metric_func(Yhat, Ytest_val)
                print(choice_value)

                if verbose > 0:
                    print(
                        "{} is {:.3f} at theta={:.3f}, percentile={:.4f}, window={}".format(
                            metric, choice_value, theta, percentile, window
                        )
                    )

                # FPR is a negative metric (lower is better)
                if negative_metric:
                    if choice_value < best_metric:
                        best_metric = choice_value
                        best_percentile = percentile
                        best_window = window
                else:
                    if choice_value > best_metric:
                        best_metric = choice_value
                        best_percentile = percentile
                        best_window = window

                if grid_config.get("save-metric-info", False):
                    metric_vals[percentile_idx, window_idx] = choice_value

                if grid_config.get("detection-plots", False):

                    fig_detect, ax_detect = plt.subplots(figsize=(20, 4))

                    ax_detect.plot(Yhat, color="0.1", label="predicted state")
                    ax_detect.plot(
                        Ytest_val, color="r", alpha=0.75, lw=2, label="real state"
                    )
                    ax_detect.fill_between(
                        np.arange(len(Yhat)), 0, Yhat.astype(int), color="0.1"
                    )
                    ax_detect.set_title(
                        "Detection trajectory on test dataset, {}, percentile={:.3f}, window={}".format(
                            model_type, percentile, window
                        ),
                        fontsize=14,
                    )
                    ax_detect.set_yticks([0, 1])
                    ax_detect.set_yticklabels(["NO ATTACK", "ATTACK"])
                    ax_detect.legend(fontsize=12, loc=2)
                    try:
                        fig_detect.savefig(
                            f"plots/{run_name}/{model_name}-{percentile}-{window}.pdf"
                        )
                        if firstPlotsError:
                            print(
                                f"Saving plots for model {model_name} to plots/{run_name}"
                            )
                            firstPlotsError = False
                    except FileNotFoundError:
                        fig_detect.savefig(
                            f"plots/results/{model_name}-{percentile}-{window}.pdf"
                        )
                        if firstPlotsError:
                            print(
                                f"Directory plots/{run_name}/ not found, saving plots for model {model_name} to plots/results/ instead"
                            )
                            firstPlotsError = False
                    plt.close(fig_detect)

            if grid_config.get("save-theta", False):
                try:
                    pickle.dump(
                        theta,
                        open(
                            f"models/{run_name}/{model_name}-{percentile}-theta.pkl",
                            "wb",
                        ),
                    )
                    print(
                        f"Saved theta to models/{run_name}/{model_name}-{percentile}-theta.pkl"
                    )
                except FileNotFoundError:
                    pickle.dump(
                        theta,
                        open(
                            f"models/results/{model_name}-{percentile}-theta.pkl", "wb"
                        ),
                    )
                    print(
                        f"Directory models/{run_name}/ not found, saved theta to models/results/{model_name}-{percentile}-theta.pkl instead"
                    )

        print(
            "Best metric ({}) is {:.3f} at percentile={:.5f}, window {}".format(
                metric, best_metric, best_percentile, best_window
            )
        )

        # Final test performance
        final_test_errors = event_detector.reconstruction_errors(
            Xtest_test, batches=do_batches
        )
        final_test_instance_errors = final_test_errors.mean(axis=1)

        best_theta = np.quantile(validation_errors.mean(axis=1), best_percentile)
        event_detector.save_detection_params(
            best_theta=best_theta, best_window=best_window
        )

        final_Yhat = event_detector.best_cached_detect(final_test_instance_errors)
        final_Yhat, Ytest_test_normalized = utils.normalize_array_length(
            final_Yhat, Ytest_test
        )

        cm = confusion_matrix(Ytest_test_normalized, final_Yhat)
        df_cm = pd.DataFrame(
            cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]
        )
        df_cm.to_csv(f"{model_name}-confusion_matrix.csv", index=True)

        print("Confusion matrix đã được lưu vào 'confusion_matrix.csv'")

        metric_func = metrics.get(metric)
        final_value = metric_func(final_Yhat, Ytest_test_normalized)
        print(
            "Final {} is {:.3f} at percentile={:.5f}, window {}".format(
                metric, final_value, best_percentile, best_window
            )
        )
        f1_micro = f1_score(Ytest_test_normalized, final_Yhat, average="micro")
        f1_macro = f1_score(Ytest_test_normalized, final_Yhat, average="macro")
        precision_micro = precision_score(
            Ytest_test_normalized, final_Yhat, average="micro"
        )
        precision_macro = precision_score(
            Ytest_test_normalized, final_Yhat, average="macro"
        )
        recall_micro = recall_score(Ytest_test_normalized, final_Yhat, average="micro")
        recall_macro = recall_score(Ytest_test_normalized, final_Yhat, average="macro")

        # Calculate total running time
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Total runtime: {runtime:.2f} seconds")

        report = classification_report(Ytest_test_normalized, final_Yhat, digits=4)
        fpr, tpr, _ = roc_curve(Ytest_test_normalized, final_Yhat)
        precision, recall, _ = precision_recall_curve(Ytest_test_normalized, final_Yhat)

        # output this to a file
        with open(f"{model_name}-classification_report.txt", "w") as f:
            f.write(report + "\n")
            f.write(f"F1 Score (micro): {f1_micro}\n")
            f.write(f"F1 Score (macro): {f1_macro}\n")
            f.write(f"Precision (micro): {precision_micro}\n")
            f.write(f"Precision (macro): {precision_macro}\n")
            f.write(f"Recall (micro): {recall_micro}\n")
            f.write(f"Recall (macro): {recall_macro}\n")
            f.write(f"Runtime: {runtime:.2f}\n")
            f.write(f"Final {metric}: {final_value}\n")
            f.write(f"Best {metric}: {best_metric}\n")
            f.write(f"Best percentile: {best_percentile}\n")
            f.write(f"Best window: {best_window}\n")
            f.write(f"Confusion matrix:\n")
            f.write(f"{df_cm}\n")
            f.write(f"ROC Curve:\n")
            f.write(f"False Positive Rate: {fpr}\n")
            f.write(f"True Positive Rate: {tpr}\n")
            f.write(f"Precision-Recall Curve:\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
        print("Classification report đã được lưu vào 'classification_report.txt'")

        # plot roc curve
        fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
        ax_roc.plot(fpr, tpr, color="b", label="ROC curve")

        ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend()

        try:
            fig_roc.savefig(f"plots/{run_name}/{model_name}-roc.pdf")
        except FileNotFoundError:
            fig_roc.savefig(f"plots/results/{model_name}-roc.pdf")
            if firstPlotsError:
                print(
                    f"Directory plots/{run_name}/ not found, saving ROC curve for model {model_name} to plots/results/ instead"
                )
                firstPlotsError = False

        plt.close(fig_roc)

        if grid_config.get("save-metric-info", False):
            try:
                np.save(f"npys/{run_name}/{model_name}-{metric}.npy", metric_vals)
                print(f"Saved metric at npys/{run_name}/{model_name}-{metric}.npy")
            except Exception as e:
                print(f"Error saving to npys/{run_name}/: {e}")
                try:
                    np.save(f"npys/results/{model_name}-{metric}.npy", metric_vals)
                    print(
                        f"Saved metric at npys/results/{model_name}-{metric}.npy instead"
                    )
                except Exception as e2:
                    print(f"Error saving metric data: {e2}. Continuing without saving.")

    return event_detector


def save_model(event_detector, config, run_name="results"):
    model_name = config["name"]
    directory = f"models/{run_name}"
    filename = f"{directory}/{model_name}"

    os.makedirs(directory, exist_ok=True)

    try:

        with open(filename + ".pkl", "wb") as f:
            pickle.dump(event_detector, f)
        print(f"Model saved at {filename}.pkl")

    except Exception as e:
        print(f"Error saving model: {e}")


def load_saved_model(model_type, run_name, model_name):
    """Load stored model."""
    if model_type == "ID":
        return identity.Identity()

    try:
        with open(f"models/{run_name}/{model_name}.json") as fd:
            model_params = json.load(fd)
        model_filename = f"models/{run_name}/{model_name}.h5"
    except FileNotFoundError:
        print(
            f"Unable to find models/{run_name}/{model_name}.json, checking models/results/..."
        )
        try:
            with open(f"models/results/{model_name}.json") as fd:
                model_params = json.load(fd)
            print(
                f"Using {model_name}.json and {model_name}.h5 found in models/results/"
            )
            print(
                "Note: we recommend separate directories to avoid writing over experiments"
            )
            model_filename = f"models/results/{model_name}.h5"
        except FileNotFoundError:
            raise SystemExit(
                f"Unable to find model {model_name}. Ensure you have trained the model first"
            )

    if model_type == "AE":
        event_detector = autoencoder.AEED(**model_params)
    elif model_type == "GRU":
        event_detector = gru.GatedRecurrentUnit(**model_params)
    elif model_type == "CNN":
        event_detector = cnn.ConvNN(**model_params)
    elif model_type == "DNN":
        event_detector = dnn.DeepNN(**model_params)
    elif model_type == "LSTM":
        event_detector = lstm.LongShortTermMemory(**model_params)
    elif model_type == "LIN":
        event_detector = linear.Linear(**model_params)
    else:
        raise SystemExit(f"Model type {model_type} is not supported.")

    # load keras model
    event_detector.inner = tf.keras.models.load_model(model_filename)

    return event_detector


def parse_arguments():

    parser = utils.get_argparser()

    parser.add_argument(
        "--rf_n_estimators",
        default=100,
        type=int,
        help="Number of trees in Random Forest (n_estimators)",
    )

    parser.add_argument(
        "--rf_model_params_n_estimators",
        default=100,
        type=int,
        help="Number of trees (n_estimators) for Random Forest model",
    )

    parser.add_argument(
        "--rf_model_params_history",
        default=100,
        type=int,
        help="History window size for Random Forest",
    )

    # # SVM parameters
    parser.add_argument(
        "--svm_model_params_C",
        default=1.0,
        type=float,
        help="Regularization parameter for SVM",
    )

    parser.add_argument(
        "--svm_model_params_kernel",
        default="rbf",
        type=str,
        choices=["linear", "poly", "rbf", "sigmoid"],
        help="Kernel type for SVM",
    )

    parser.add_argument(
        "--svm_model_params_history",
        default=50,
        type=int,
        help="History window size for SVM",
    )

    # KNN parameters
    parser.add_argument(
        "--knn_model_params_n_neighbors",
        default=5,
        type=int,
        help="Number of neighbors for KNN",
    )

    parser.add_argument(
        "--knn_model_params_weights",
        default="uniform",
        type=str,
        choices=["uniform", "distance"],
        help="Weight function for KNN",
    )

    parser.add_argument(
        "--knn_model_params_history",
        default=50,
        type=int,
        help="History window size for KNN",
    )

    # MLP parameters
    parser.add_argument(
        "--mlp_model_params_hidden_layer_sizes",
        default="100,50",
        type=str,
        help="Hidden layer sizes for MLP (comma-separated)",
    )

    parser.add_argument(
        "--mlp_model_params_activation",
        default="relu",
        type=str,
        choices=["identity", "logistic", "tanh", "relu"],
        help="Activation function for MLP",
    )

    parser.add_argument(
        "--mlp_model_params_max_iter",
        default=200,
        type=int,
        help="Maximum iterations for MLP",
    )

    parser.add_argument(
        "--mlp_model_params_history",
        default=50,
        type=int,
        help="History window size for MLP",
    )

    ### Train Params
    parser.add_argument(
        "--train_params_epochs", default=100, type=int, help="Number of training epochs"
    )
    parser.add_argument(
        "--train_params_batch_size",
        default=512,
        type=int,
        help="Training batch size. Note: MUST be larger than history/window values given",
    )
    parser.add_argument(
        "--train_params_no_callbacks",
        action="store_true",
        help="Remove callbacks like early stopping",
    )

    # Hyperparameters
    parser.add_argument(
        "--detect_params_percentile",
        default=[
            0.95,
            0.96,
            0.97,
            0.98,
            0.99,
            0.991,
            0.992,
            0.993,
            0.994,
            0.995,
            0.996,
            0.997,
            0.998,
            0.999,
            0.9995,
            0.99995,
        ],
        nargs="+",
        type=float,
        help="Percentiles to look over",
    )
    parser.add_argument(
        "--detect_params_windows",
        default=[1, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        nargs="+",
        type=int,
        help="Windows to look over",
    )
    parser.add_argument(
        "--detect_params_metrics",
        default=["F1"],
        nargs="+",
        type=str,
        help="Metrics to look over",
    )
    parser.add_argument(
        "--detect_params_test_split",
        default=0.7,
        type=float,
        help="Split for testing/validation of detection hyperparameters. Default is 0.7 (hyperparameters evaluated on 30%% of test data, final testing on 70%%.) ",
    )

    # saving items
    parser.add_argument(
        "--detect_params_plots",
        action="store_true",
        help="Make detection plots for each hyperparameter setting",
    )
    parser.add_argument(
        "--detect_params_save_npy",
        action="store_true",
        help="Save the metric values in an npy",
    )
    parser.add_argument(
        "--detect_params_save_theta",
        action="store_true",
        help="Save theta thresholds in a pkl",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    model_type = args.model
    dataset_name = args.dataset

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

    # Define training parameters
    ae_train_params = {
        "verbose": 1,
        "batch_size": args.train_params_batch_size,
        "epochs": args.train_params_epochs,
        "use_callbacks": not args.train_params_no_callbacks,
    }

    large_train_params = {
        "batch_size": args.train_params_batch_size,
        "epochs": args.train_params_epochs,
        "use_callbacks": not args.train_params_no_callbacks,
        "steps_per_epoch": 0,
        "validation_steps": 0,
        "verbose": 1,
    }

    config = {
        "grid_search": {
            "percentile": args.detect_params_percentile,
            "window": args.detect_params_windows,
            "metrics": args.detect_params_metrics,
            "pr-plot": False,
            "detection-plots": args.detect_params_plots,
            "save-metric-info": args.detect_params_save_npy,
            "save-theta": args.detect_params_save_theta,
        }
    }

    run_name = args.run_name
    test_split = args.detect_params_test_split
    print(args)
    utils.update_config_model(args, config, model_type, dataset_name)
    print("NAme  ", config["name"])
    model_name = config["name"]
    print("Model name", model_name)

    Xfull, sensor_cols = load_train_data(dataset_name)

    Xtest, Ytest, _ = load_test_data(dataset_name)
    print("Xfull shape: ", Xfull.shape)
    print("Xtest shape: ", Xtest.shape)
    print("Ytest shape: ", Ytest.shape)

    # X_train_windowed, Y_train_windowed = utils.transform_to_window_data(Xfull, Xfull, history=100)
    # X_test_windowed, Y_test_windowed = utils.transform_to_window_data(Xtest, Ytest, history=100)

    # print("X_train_windowed shape: ", X_train_windowed.shape)
    # print("Y_train_windowed shape: ", Y_train_windowed.shape)
    # print("X_test_windowed shape: ", X_test_windowed.shape)
    # print("Y_test_windowed shape: ", Y_test_windowed.shape)

    shuffle = True
    by_idx = True

    model_params = config["model"]

    # Updates training parameters such as batch size, learning rate, etc.
    if model_type == "AE":
        config.update({"train": ae_train_params})
        Xtrain, Xval, _, _ = train_test_split(
            Xfull, Xfull, test_size=0.2, random_state=42, shuffle=False
        )
        event_detector = train_reconstruction_model(model_type, config, Xtrain, Xval)

        # Search for the best tuning of the window and theta parameters
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
            verbose=0,
        )

    elif model_type in ["RF", "XG", "SVM", "KNN", "MLP"]:  # Check for all ML algorithms
        print(f"Training {model_type} Model...")
        Xtrain, Xval, _, _ = train_test_split(
            Xfull, Xfull, test_size=0.2, random_state=42, shuffle=False
        )
        event_detector = train_ml_model(model_type, config, Xtrain, Xval)

        # Perform hyperparameter search
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

    else:

        history = config["model"]["history"]

        train_idxs, val_idxs = utils.train_val_history_idx_split(Xfull, history)

        large_train_params["steps_per_epoch"] = (
            len(train_idxs) // large_train_params["batch_size"]
        )
        large_train_params["validation_steps"] = (
            len(val_idxs) // large_train_params["batch_size"]
        )
        config.update({"train": large_train_params})

        event_detector = train_forecast_model_by_idxs(
            model_type, config, Xfull, train_idxs, val_idxs
        )
        print("-----------------------------------------------")
        print(val_idxs)
        # Search for the best tuning of the window and theta parameters
        hyperparameter_search(
            event_detector,
            model_type,
            config,
            Xfull,
            Xtest,
            Ytest,
            dataset_name,
            val_idxs=val_idxs,
            test_split=test_split,
            run_name=run_name,
            verbose=0,
        )

    save_model(event_detector, config, run_name=run_name)

    print("Finished!")
