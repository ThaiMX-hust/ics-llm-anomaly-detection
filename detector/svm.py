import json
import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from .detector import ICSDetector
import pickle


class SVMRegressor(ICSDetector):
    def __init__(self, **kwargs):
        params = {
            "C": 1.0,
            "kernel": "rbf",
            "epsilon": 0.1,
            "gamma": "scale",
            "random_state": 42,
            "history": 50,
            "verbose": 1,
        }

        for key, item in kwargs.items():
            params[key] = item

        self.params = params
        self.scaler = StandardScaler()
        base_svr = SVR(
            C=self.params["C"],
            kernel=self.params["kernel"],
            epsilon=self.params["epsilon"],
            gamma=self.params["gamma"],
        )
        self.model = MultiOutputRegressor(base_svr)
        self.is_multioutput = True

    def transform_to_window_data(self, dataset, target, target_size=1):
        data, labels = [], []
        history = self.params["history"]
        target_is_multidim = len(target.shape) > 1 and target.shape[1] > 1
        is_reconstruction = np.array_equal(dataset, target)

        for i in range(history, len(dataset) - target_size):
            window_data = dataset[i - history : i].flatten()
            data.append(window_data)
            if is_reconstruction:
                labels.append(dataset[i + target_size])
            else:
                if target_is_multidim:
                    labels.append(target[i + target_size])
                else:
                    labels.append(target[i + target_size])

        return np.array(data), np.array(labels)

    def train(self, Xtrain, Ytrain):
        if self.params["verbose"]:
            print("Starting SVM training...")
        is_reconstruction = len(Ytrain.shape) > 1 and Ytrain.shape[1] > 1
        if is_reconstruction:
            print(
                f"Using mean-target approach for multidimensional target with shape {Ytrain.shape}"
            )
            self.is_multioutput = False
            self.model = SVR(
                C=self.params["C"],
                kernel=self.params["kernel"],
                epsilon=self.params["epsilon"],
                gamma=self.params["gamma"],
            )
            Ytrain_mean = np.mean(Ytrain, axis=1)

            self.scaler = StandardScaler()
            Xtrain_scaled = self.scaler.fit_transform(Xtrain)
            self.model.fit(Xtrain_scaled, Ytrain_mean)

            if self.params["verbose"]:
                print(
                    f"Trained SVR model on {Xtrain.shape[0]} samples with input shape {Xtrain.shape[1]}"
                )
        else:
            try:
                self.scaler = StandardScaler()
                Xtrain_scaled = self.scaler.fit_transform(Xtrain)
                self.model.fit(Xtrain_scaled, Ytrain)

                if self.params["verbose"]:
                    print(
                        f"Trained MultiOutputRegressor with input shape {Xtrain.shape}"
                    )
            except (ValueError, TypeError) as e:
                print(f"Error in multioutput training: {e}")
                print("Falling back to single-output SVR model...")

                self.is_multioutput = False
                self.model = SVR(
                    C=self.params["C"],
                    kernel=self.params["kernel"],
                    epsilon=self.params["epsilon"],
                    gamma=self.params["gamma"],
                )
                if len(Ytrain.shape) > 1 and Ytrain.shape[1] > 1:
                    Ytrain_mean = np.mean(Ytrain, axis=1)
                else:
                    Ytrain_mean = Ytrain
                Xtrain_scaled = self.scaler.fit_transform(Xtrain)
                self.model.fit(Xtrain_scaled, Ytrain_mean)

        if self.params["verbose"]:
            print("SVM training completed.")

    def predict(self, X):
        try:
            if len(X.shape) > 2:
                X_2d = X.reshape(X.shape[0], -1)
            else:
                X_2d = X
            X_scaled = self.scaler.transform(X_2d)
            predictions = self.model.predict(X_scaled)

            return predictions
        except ValueError as e:
            print(f"Prediction error: {e}")
            print("Original X shape:", X.shape)
            print("Using alternative approach...")
            X_flat = X.flatten().reshape(1, -1)
            try:
                X_scaled = self.scaler.transform(X_flat)
                pred = self.model.predict(X_scaled)

                if len(X.shape) > 1:
                    result = np.full(X.shape[0], pred[0])
                    return result
                return pred
            except Exception as e2:
                print(f"Alternative prediction also failed: {e2}")
                print("Returning zeros as predictions")
                if len(X.shape) > 1:
                    return np.zeros(X.shape[0])
                return np.zeros(1)

    def detect(self, x, theta, window=1):
        reconstruction_error = self.reconstruction_errors(x)
        instance_errors = reconstruction_error.mean(axis=1)
        return self.cached_detect(instance_errors, theta, window)

    def cached_detect(self, instance_errors, theta, window=1):
        detection = instance_errors > theta
        if window > 1:
            detection = np.convolve(detection, np.ones(window), "same") // window
        return detection

    def reconstruction_errors(self, x, batches=False):
        Xwindow, Ywindow = self.transform_to_window_data(x, x)
        if self.is_multioutput == False:
            X_scaled = self.scaler.transform(Xwindow)
            predictions = self.model.predict(X_scaled)
            actual_values = np.mean(Ywindow, axis=1)
            squared_errors = (predictions - actual_values) ** 2
            expanded_errors = np.zeros(Ywindow.shape)
            for i in range(Ywindow.shape[1]):
                expanded_errors[:, i] = squared_errors

            return expanded_errors
        else:
            try:
                X_scaled = self.scaler.transform(Xwindow)
                predictions = self.model.predict(X_scaled)
                if predictions.shape != Ywindow.shape:
                    try:
                        predictions = predictions.reshape(Ywindow.shape)
                    except ValueError:
                        print(
                            "Warning: Shape mismatch, using mean approach as fallback"
                        )
                        predictions_flat = predictions.flatten()
                        predictions_mean = np.mean(predictions_flat)
                        predictions = np.full(Ywindow.shape, predictions_mean)

                return (predictions - Ywindow) ** 2

            except ValueError as e:
                print(f"Warning: {e}. Falling back to mean-based approach.")
                self.is_multioutput = False
                return self.reconstruction_errors(x, batches)

    def save(self, filename):
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "is_multioutput": self.is_multioutput,
                },
                f,
            )
        print(f" SVM model saved at {filename}.pkl")

    def load(self, filename):
        with open(filename + ".pkl", "rb") as f:
            saved_data = pickle.load(f)
            self.model = saved_data["model"]
            self.scaler = saved_data["scaler"]
            self.is_multioutput = saved_data.get("is_multioutput", True)
        print(f" SVM model loaded from {filename}.pkl")


if __name__ == "__main__":
    print("Not a main file.")
