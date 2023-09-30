import numpy as np

class LinearRegression:
    def __init__(self):
        self.w = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.w = np.linalg.inv(X.T @ X) @ (X.T @ y)  

    def predict(self, X: np.ndarray)->np.ndarray:
        return X @ self.w
    
    def compute_cost(self, X: np.ndarray, y: np.ndarray)->np.float64:
        y_hat = X @ self.w
        squared_error = 0.5 * (y - y_hat) ** 2
        return squared_error