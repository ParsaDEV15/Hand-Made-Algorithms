from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np


class LogisticRegressionMBGD:
    def __init__(self, n_iter=1000, alpha=0.0, learning_rate=0.001, random_state=None):
        self.rng = np.random.default_rng(random_state)

        self.n_iter = n_iter
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.batch_size = 32

        self.w = None
        self.b = None

    def fit(self, X, y):
        number_of_samples = X.shape[0]
        number_of_features = X.shape[1]

        self.w = np.random.normal(0, 0.02, size=number_of_features)
        self.b = np.random.normal(0, 0.02, size=1)[0]

        for _ in range(self.n_iter):
            shuffled_indexes = self.rng.permutation(number_of_samples)

            shuffled_X = X[shuffled_indexes]
            shuffled_y = y[shuffled_indexes]

            for i in range(0, number_of_samples, self.batch_size):
                batch_X = shuffled_X[i:i + self.batch_size]
                batch_y = shuffled_y[i:i + self.batch_size]

                batch_size = batch_y.shape[0]

                z = batch_X @ self.w + self.b
                y_hat_batch = 1 / (1 + np.exp(-z))

                error = y_hat_batch - batch_y

                d_w = ((batch_X.T @ error) / batch_size) + (self.alpha * self.w)
                d_b = np.sum(error) / batch_size

                updated_w = self.w - self.learning_rate * d_w
                updated_b = self.b - self.learning_rate * d_b

                self.w = updated_w
                self.b = updated_b

    def predict(self, X):
        z = X @ self.w + self.b
        sigmoid = 1 / (1 + np.exp(-z))

        return (sigmoid >= 0.5).astype(int)


if __name__ == '__main__':
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    lr = LogisticRegressionMBGD()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    print(classification_report(y_test, y_pred))