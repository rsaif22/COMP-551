import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split


class LinearRegressionSGD:
    def __init__(self, D):
        self.w = np.zeros((D, 1))

    def compute_gradient(self, X: np.ndarray, y: np.ndarray):
        N = y.shape[0]
        y_hat = X @ self.w
        return 1/N * X.T @ (y_hat-y)
    
    def compute_cost(self, X: np.ndarray, y: np.ndarray)->np.float64:
        y_hat = X @ self.w
        squared_error = np.sum(0.5 * (y - y_hat) ** 2)
        return squared_error
    
    def fit(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, epsilon: float = 1e-6,
            batch_size: int = 10, max_iters = 1e8):
        y = np.reshape(y, (y.size, 1))
        grad = self.compute_gradient(X, y)
        num_iters = 0
        while (np.linalg.norm(grad) > epsilon and num_iters < max_iters):
            cost = self.compute_cost(X, y)
            print(cost)
            self.w = self.w - learning_rate*grad
            grad = self.compute_gradient(X, y)
            num_iters += 1

    def predict(self, X: np.ndarray):
        return X @ self.w
    
def normalize(X: np.ndarray):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

def add_bias(X: np.ndarray):
    return np.column_stack((X, np.ones((X.shape[0], 1))))

if __name__=="__main__":
    boston_df = None

    with open("housing.csv", "r") as f:
        names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD",
                            "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
        boston_df = pd.read_csv(f, header=None, index_col=False, names=names, sep=r'\s+') # Space separated csv

    boston_df = boston_df.drop(["B"], axis=1) # Remove unethical data
    linear_reg_sgd_model = LinearRegressionSGD(boston_df.shape[1])
    
    boston_np = boston_df.to_numpy()
    boston_X = boston_np[:, :-1]
    boston_y = boston_np[:, -1]

    boston_X = normalize(boston_X)
    boston_X = add_bias(boston_X)
    X_train, X_test, y_train, y_test = train_test_split(boston_X, boston_y, test_size=0.2, shuffle=True)

    linear_reg_sgd_model.fit(X_train, y_train)

    y_test_hat = linear_reg_sgd_model.predict(X_test)

    y_comp = np.column_stack((y_test, y_test_hat))
    test_y_df = pd.DataFrame(y_comp, columns=["True y", "Predicted y"])
    print(test_y_df)
