from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from models.logistic_regression import LogisticRegressionMBGD


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