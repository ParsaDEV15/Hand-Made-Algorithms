from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


class LinearRegressionSGD:
    def __init__(self, n_iter=100, alpha=0.0, learning_rate=0.001, random_state=None):
        self.rng = np.random.default_rng(random_state)

        self.n_iter = n_iter
        self.alpha = alpha
        self.learning_rate = learning_rate

        self.w = None
        self.b = None

    def fit(self, X, y):
        number_of_samples = X.shape[0]
        number_of_features = X.shape[1]

        self.w = np.random.normal(0, 0.02, size=number_of_features)
        self.b = np.random.normal(0, 0.02, size=1)[0]

        for _ in range(self.n_iter):
            rand_index = self.rng.integers(number_of_samples, size=1)[0]
            X_i, y_i = X[rand_index], y[rand_index]

            y_hat = X_i @ self.w + self.b

            error = y_i - y_hat
            d_w = -2 * (X_i * error) + (self.alpha * self.w)
            d_b = -2 * error

            updated_w = self.w - self.learning_rate * d_w
            updated_b = self.b - self.learning_rate * d_b

            self.w, self.b = updated_w, updated_b

    def predict(self, X):
        return X @ self.w + self.b


if __name__ == '__main__':
    X, y = make_regression(n_samples=100, n_features=5, noise=15, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    lr = LinearRegressionSGD(
        n_iter=5000,
        alpha=0.001,
        random_state=42,
        learning_rate=0.001
    )
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print('MSE:', mse)
    print('MAE:', mae)