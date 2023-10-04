import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split


class LinearRegressionSGD:
    def __init__(self, D):
        self.w = np.zeros((D+1, 1))
        self.mean = np.zeros((D,))
        self.std =  np.zeros((D,))
        self.error_array = np.array([0])

    def compute_gradient(self, X: np.ndarray, y: np.ndarray):
        N = y.shape[0]
        y_hat = X @ self.w
        return 1/N * X.T @ (y_hat-y)
    
    def compute_error(self, X: np.ndarray, y: np.ndarray)->np.float64:
        y_hat = self.predict(X)
        mean_squared_error = np.mean(0.5 * (y - y_hat) ** 2)
        return mean_squared_error
    

    def fit(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, epsilon: float = 1e-4,
            batch_size: int = 15, max_iters = 1e4):
        # Data already shuffled
        self.error_array = np.empty((0,))
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X_normalized = self.normalize(X)
        y = np.reshape(y, (y.size, 1))
        all_indices = np.arange(y.size) # All n values
        current_batch = np.random.choice(all_indices, batch_size)
        grad = self.compute_gradient(X_normalized[current_batch, :], y[current_batch])
        num_iters = 0
        while (np.linalg.norm(grad) > epsilon and num_iters < max_iters):
            #print(self.compute_error(X, y))
            self.error_array = np.append(self.error_array, self.compute_error(X, y))
            self.w = self.w - learning_rate * grad
            current_batch = np.random.choice(all_indices, batch_size)
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

if __name__=="__main__":
    boston_df = None

    with open("housing.csv", "r") as f:
        names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD",
                            "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
        boston_df = pd.read_csv(f, header=None, index_col=False, names=names, sep=r'\s+') # Space separated csv

    boston_df = boston_df.drop(["B"], axis=1) # Remove unethical data
   
    
    
    #linear_reg_sgd_model.fit(X_train, y_train)
    # linear_reg_sgd_model.fit(X_train, y_train)

    # y_test_hat = linear_reg_sgd_model.predict(X_test)

    # y_comp = np.column_stack((y_test, y_test_hat))
    # test_y_df = pd.DataFrame(y_comp, columns=["True y", "Predicted y"])
    # #print(test_y_df)
    # #print(linear_reg_sgd_model.compute_error(X_test, y_test))
    # print(linear_reg_sgd_model.w)

