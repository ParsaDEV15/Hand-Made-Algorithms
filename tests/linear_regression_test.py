from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.linear_regression_sgd import LinearRegressionSGD


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