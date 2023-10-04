import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


class LinearRegression:
    def __init__(self, D: int):
        self.w = np.zeros((D + 1, 1))
        self.mean = np.zeros((D,))
        self.std = np.zeros((D,))

    # ADD BIAS TERMS

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X_normalized = self.normalize(X)
        self.w = np.linalg.inv(X_normalized.T @ X_normalized) @ (X_normalized.T @ y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_normalized = self.normalize(X)
        return X_normalized @ self.w

    def compute_error(self, X: np.ndarray, y: np.ndarray) -> np.float64:
        y_hat = self.predict(X)
        mean_squared_error = np.mean(0.5 * (y - y_hat) ** 2)
        return mean_squared_error

    def normalize(self, X: np.ndarray, add_bias: bool = True):
        X_normalized = (X - self.mean) / self.std
        if add_bias:
            X_normalized = np.column_stack((X_normalized, np.ones((X_normalized.shape[0], 1))))
        return X_normalized


if __name__ == "__main__":
    boston_df = None

    with open("housing.csv", "r") as f:
        names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD",
                 "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
        boston_df = pd.read_csv(f, header=None, index_col=False, names=names, sep=r'\s+')  # Space separated csv

    boston_df = boston_df.drop(["B"], axis=1)  # Remove unethical data

    boston_np = boston_df.to_numpy()
    boston_X = boston_np[:, :-1]
    boston_y = boston_np[:, -1]
    # print(boston_X)

    linear_reg_model = LinearRegression(boston_X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(boston_X, boston_y, test_size=0.2, shuffle=True)

    linear_reg_model.fit(X_train, y_train)

    #y_test_hat = linear_reg_model.predict(X_test)

    error_without_gaussian = linear_reg_model.compute_error(X_test, y_test)
    print("Mean Squared Error without Gaussian features:", error_without_gaussian)


    # Gaussian function

    def calculate_gaussian_basis_functions(X, centers, s):
        num_samples = X.shape[0]
        num_centers = centers.shape[0]
        basis_functions = np.zeros((num_samples, num_centers))

        for j in range(num_centers):
            distances = np.sum((X - centers[j]) ** 2, axis=1)
            phi_j = np.exp(-distances / (2 * s ** 2))
            basis_functions[:, j] = phi_j

        return basis_functions


    num_basis_functions = 5
    s = 1

    # Randomly select µj values from the training set to determine basis function centers
    random_indices = np.random.choice(X_train.shape[0], num_basis_functions, replace=False)
    basis_function_centers = X_train[random_indices]
    #print(random_indices, basis_function_centers, basis_function_centers.shape[0])

    #print(X_train)
    #print("middle")

    # Calculate Gaussian basis functions for the enriched feature set
    X_train_enriched = calculate_gaussian_basis_functions(X_train, basis_function_centers, s)
    X_test_enriched = calculate_gaussian_basis_functions(X_test, basis_function_centers, s)

    #print(X_test_enriched)
    # Rebuild model for the enriched feature set
    linear_reg_model_gaussian = LinearRegression(X_train_enriched.shape[1])

    # Train on enriched data
    linear_reg_model_gaussian.fit(X_train_enriched, y_train)

    # Get error for the model with Gaussian features
    error_with_gaussian = linear_reg_model_gaussian.compute_error(X_test_enriched, y_test)
    print("Mean Squared Error with Gaussian features:", error_with_gaussian)
