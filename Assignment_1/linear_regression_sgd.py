import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
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
        self.time_array = np.empty((0, ))
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
        print(f"Batch size: {batch_size}, converged in: {num_iters} iterations")

    def predict(self, X: np.ndarray):
        X_normalized = self.normalize(X)
        return X_normalized @ self.w
    
    def normalize(self, X: np.ndarray, add_bias: bool = True):
        X_normalized = (X - self.mean) / self.std
        if add_bias:
            X_normalized = np.column_stack((X_normalized, np.ones((X_normalized.shape[0], 1))))
        return X_normalized

if __name__=="__main__":
    boston_df = None

    with open("Assignment_1/housing.csv", "r") as f:
        names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD",
                            "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
        boston_df = pd.read_csv(f, header=None, index_col=False, names=names, sep=r'\s+') # Space separated csv

    boston_df = boston_df.drop(["B"], axis=1) # Remove unethical data
   
    
    import matplotlib.pyplot as plt
    boston_np = boston_df.to_numpy()
    boston_X = boston_np[:, :-1]
    boston_y = boston_np[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(boston_X, boston_y, test_size=0.2)
    linear_sgd_model = LinearRegressionSGD(X_train.shape[1])
    linear_sgd_model.fit(X_train, y_train, batch_size=y_train.size, epsilon=1e-3, max_iters=1e4, learning_rate=0.01)
    base_case = linear_sgd_model.error_array
    base_time = linear_sgd_model.time_array
    test_batch_sizes = [8, 16, 32, 64, 128, 256, 400]
    result_dict = {"Full batch":linear_sgd_model.compute_error(X_test, y_test)}
    for size in test_batch_sizes:
        plt.figure()
        linear_sgd_model.fit(X_train, y_train, batch_size=size, epsilon=1e-3, max_iters=1e4, learning_rate=0.01)
        error_array = linear_sgd_model.error_array
        time_array = linear_sgd_model.time_array
        result_dict[f"Size {size}"] = linear_sgd_model.compute_error(X_test, y_test)
        plt.plot(base_time[:100], base_case[:100], label="Full batch")
        plt.plot(time_array[:100], error_array[:100], label=f"Batch size: {size}")
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("MSE")
        plt.title("Effect of Batch Size")
        plt.show()

