import json
import numpy as np
import pdb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from .detector import ICSDetector
import pickle


class RFRegressor(ICSDetector):
    def __init__(self, **kwargs):
        params = {
            "n_estimators": 50,
            "max_depth": None,
            "random_state": 42,
            "history": 50,
            "verbose": 1,
        }

        for key, item in kwargs.items():
            params[key] = item

        self.params = params
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            random_state=self.params["random_state"],
            verbose=self.params["verbose"],
        )

    def transform_to_window_data(self, dataset, target, target_size=1):
        data, labels = [], []
        history = self.params["history"]
        target_is_multidim = len(target.shape) > 1 and target.shape[1] > 1

        is_reconstruction = np.array_equal(dataset, target)

        for i in range(history, len(dataset) - target_size):
            data.append(dataset[i - history : i])

            # For reconstruction tasks, use the next timestep of dataset
            if is_reconstruction:
                labels.append(dataset[i + target_size])
            # For multidimensional target, use mean of all features for more comprehensive representation
            elif target_is_multidim:
                # Use mean of all features instead of just first column
                labels.append(np.mean(target[i + target_size]))
            else:
                labels.append(target[i + target_size])

        return np.array(data), np.array(labels)

    def train(self, Xtrain, Ytrain):
        if self.params["verbose"]:
            print("Starting training...")

        Xtrain = Xtrain.reshape(Xtrain.shape[0], -1)
        Xtrain = self.scaler.fit_transform(Xtrain)
        # Small data
        # X_train_small = Xtrain[0:10]
        # Y_train_small = Ytrain[0:10]
        # self.model.fit(X_train_small, Y_train_small)

        self.model.fit(Xtrain, Ytrain)

        if self.params["verbose"]:
            print("Training completed.")

    def predict(self, X):
        X = X.reshape(X.shape[0], -1)
        X = self.scaler.transform(X)

        return self.model.predict(X)

    def detect(self, x, theta, window=1):
        reconstruction_error = self.reconstruction_errors(x)
        instance_errors = reconstruction_error.mean(axis=1)
        return self.cached_detect(instance_errors, theta, window)

    def cached_detect(self, instance_errors, theta, window=1):
        """Uses precomputed errors for detection."""
        detection = instance_errors > theta
        if window > 1:
            detection = np.convolve(detection, np.ones(window), "same") // window
        return detection

    def reconstruction_errors(self, x, batches=False):
        """Computes reconstruction errors."""
        Xwindow, Ywindow = self.transform_to_window_data(x, x)

        predictions = self.predict(Xwindow)

        if len(Ywindow.shape) > 1 and len(predictions.shape) == 1:
            actual_values = np.mean(Ywindow, axis=1)
            squared_errors = (predictions - actual_values) ** 2

            expanded_errors = np.zeros(Ywindow.shape)
            for i in range(Ywindow.shape[1]):
                expanded_errors[:, i] = squared_errors

            return expanded_errors
        elif predictions.shape != Ywindow.shape:
            raise ValueError(
                f"Prediction shape {predictions.shape} does not match Ywindow shape {Ywindow.shape}"
            )
        else:
            return (predictions - Ywindow) ** 2

    def save(self, filename):
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(self.model, f)
        print(f" RF model saved at {filename}.pkl")

    def load(self, filename):
        with open(filename + ".pkl", "rb") as f:
            self.model = pickle.load(f)
        print(f" RF model loaded from {filename}.pkl")


if __name__ == "__main__":
    print("Not a main file.")
