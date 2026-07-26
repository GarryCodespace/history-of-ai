import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = 0.0

    def weighted_sum(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def activation_function(self, weighted_sum):
        predictions = (np.asarray(weighted_sum) >= 0).astype(int)
        return int(predictions) if predictions.ndim == 0 else predictions

    def predict(self, inputs):
        weighted_sum = self.weighted_sum(inputs)
        return self.activation_function(weighted_sum)

    def fit(self, X, y):
        _, number_of_features = X.shape
        self.weights = np.zeros(number_of_features)
        self.bias = 0.0

        for epoch in range(self.epochs):
            total_error = 0

            for inputs, label in zip(X, y):
                prediction = self.predict(inputs)
                error = label - prediction

                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

                total_error += abs(error)

            print(
                f"Epoch {epoch + 1}: "
                f"errors={total_error}, "
                f"weights={self.weights}, "
                f"bias={self.bias}"
            )
            if total_error == 0:
                break

        return self


if __name__ == "__main__":
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    y = np.array([0, 0, 0, 1])

    model = Perceptron()
    model.fit(X, y)
    print(f"Predictions: {model.predict(X)}")
