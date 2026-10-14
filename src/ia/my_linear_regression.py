import numpy as np


class MyLinearRegression:

    def __init__(self):
        self.b_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]

        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

        self.intercept_ = theta[0]
        self.b_ = theta[1:]

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.b_

    def mse(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot


if __name__ == "__main__":
    X = [[1], [2], [3], [4], [5]]
    y = [3, 5, 7, 9, 11]

    model = MyLinearRegression().fit(X, y)
    predictions = model.predict(X)

    print(f"Intercepto: {model.intercept_:.2f}")
    print(f"Coeficientes: {model.b_}")
    print(f"Predicciones: {predictions}")
    print(f"MSE: {model.mse(y, predictions):.2f}")
    print(f"R²: {model.r2_score(y, predictions):.2f}")