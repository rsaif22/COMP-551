import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd 

class LinearRegression:
    def __init__(self, D: int):
        self.w = np.zeros((D + 1, 1))
        self.mean = np.zeros((D,))
        self.std =  np.zeros((D,))

    # ADD BIAS TERMS
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X_normalized = self.normalize(X)
        self.w = np.linalg.inv(X_normalized.T @ X_normalized) @ (X_normalized.T @ y)  

    def compute_error(self, X: np.ndarray, y: np.ndarray)->np.float64:
        y_hat = self.predict(X)
        y_hat = np.reshape(y_hat, (y_hat.size, 1))
        y = np.reshape(y, (y.size, 1))
        mean_squared_error = np.mean(0.5 * (y - y_hat) ** 2)
        return mean_squared_error
    
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
    
    
    boston_np = boston_df.to_numpy()
    boston_X = boston_np[:, :-1]
    boston_y = boston_np[:, -1]

    linear_reg_model = LinearRegression(boston_X.shape[1])

    # boston_X = normalize(boston_X)
    # boston_X = add_bias(boston_X)
    X_train, X_test, y_train, y_test = train_test_split(boston_X, boston_y, test_size=0.2, shuffle=True)
    
    linear_reg_model.fit(X_train, y_train)

    y_test_hat = linear_reg_model.predict(X_test)

    y_comp = np.column_stack((y_test, y_test_hat))
    test_y_df = pd.DataFrame(y_comp, columns=["True y", "Predicted y"])
    #print(test_y_df)
    print(linear_reg_model.compute_error(X_test, y_test))
    #print(linear_reg_model.compute_error(X_test, y_test))
    #print(linear_reg_model.w)