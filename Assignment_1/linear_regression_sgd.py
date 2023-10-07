import numpy as np 
import pandas as pd
import time

class LinearRegressionSGD:
    def __init__(self, D):
        self.w = np.zeros((D+1, 1))
        self.mean = np.zeros((D,))
        self.std =  np.zeros((D,))
        self.error_array = np.empty((0,))
        self.time_array = np.empty((0,))

    def compute_gradient(self, X: np.ndarray, y: np.ndarray):
        N = y.shape[0]
        y_hat = X @ self.w
        return 1/N * X.T @ (y_hat-y)
    
    def compute_error(self, X: np.ndarray, y: np.ndarray)->np.float64:
        y_hat = self.predict(X)
        y_hat = np.reshape(y_hat, (y_hat.size, 1))
        y = np.reshape(y, (y.size, 1))
        mean_squared_error = np.mean(0.5 * (y - y_hat) ** 2)
        return mean_squared_error
    

    def fit(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, epsilon: float = 1e-4,
            batch_size: int = 15, max_iters = 1e4):
        # Data already shuffled
        self.w = np.zeros(self.w.shape)
        self.error_array = np.empty((0,))
        self.time_array = np.empty((0,))
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X_normalized = self.normalize(X)
        y = np.reshape(y, (y.size, 1))
        all_indices = np.arange(y.size) # All n values
        current_batch = np.random.choice(all_indices, batch_size)
        grad = self.compute_gradient(X_normalized[current_batch, :], y[current_batch])
        num_iters = 0
        time_start = time.time()
        while (np.linalg.norm(grad) > epsilon and num_iters < max_iters):
            #print(self.compute_error(X, y))
            self.error_array = np.append(self.error_array, self.compute_error(X, y))
            time_from_start = time.time() - time_start
            self.time_array = np.append(self.time_array, time_from_start)
            self.w = self.w - learning_rate * grad
            current_batch = np.random.choice(all_indices, batch_size, replace=False)
            grad = self.compute_gradient(X_normalized[current_batch, :], y[current_batch])
            num_iters += 1

    def predict(self, X: np.ndarray):
        X_normalized = self.normalize(X)
        return X_normalized @ self.w
    
    def normalize(self, X: np.ndarray, add_bias: bool = True):
        X_normalized = (X - self.mean) / self.std
        if add_bias:
            X_normalized = np.column_stack((X_normalized, np.ones((X_normalized.shape[0], 1))))
        return X_normalized


