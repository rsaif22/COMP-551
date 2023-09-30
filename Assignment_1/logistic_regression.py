import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, D: int, num_classes: int):
        self.w = np.zeros((D, num_classes)) # column for each class
        self.num_classes = num_classes

    def logistic(self, z: np.ndarray):
        return 1/(1 + np.exp(-z))
    
    def softmax(self, z:np.ndarray):
        softmax_matrix = np.zeros(z.shape)
        for i in range(z.shape[1]):
            softmax_matrix[:, i] = np.exp(z[:, i]) / np.sum(np.exp(z), axis=1)
        return softmax_matrix

    def compute_gradient(self, X: np.ndarray, y: np.ndarray):
        N = y.shape[0]
        y_hat = self.softmax(X @ self.w)
        #for i in range()
        return 1/N * X.T @ (y_hat-y)
    
    def normalize_features(self, X):
        mean = np.mean(X, axis=0) # N X D matrix
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / std
        return X_normalized
    
    def encode_y(self, y):
        encoded_y = np.broadcast_to(y, (y.size, self.num_classes)).copy()
        for i in range(self.num_classes):
            class_num = i + 1 # Not zero indexed
            encoded_y[:, i] = encoded_y[:, i] == class_num
        return encoded_y
            

    def fit(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, epsilon: float = 1e-4):
        X_normalized = self.normalize_features(X)
        y_encoded = self.encode_y(y)
        grad = self.compute_gradient(X_normalized, y_encoded)
        num_iters = 0
        while(np.linalg.norm(grad) > epsilon and num_iters < 1e4):
            self.w = self.w - learning_rate*grad
            grad = self.compute_gradient(X_normalized, y_encoded)
            num_iters += 1
    
    def predict(self, X):
        X_normalized = self.normalize_features(X)
        y_probs =  self.softmax(X_normalized @ self.w)
        y_hat = np.argmax(y_probs, axis=1) + 1
        return y_hat

if __name__=="__main__":
    from ucimlrepo import fetch_ucirepo 
  
    # fetch dataset 
    wine = fetch_ucirepo(id=109) 
    
    # data (as pandas dataframes) 
    X = wine.data.features 
    y = wine.data.targets 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    logistic_reg = LogisticRegression(X.shape[1], 3)
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()
    logistic_reg.fit(X_train_np, y_train_np)

    y_hat_test = logistic_reg.predict(X_test.to_numpy())
    #print(y_hat_test)
    y_comp = np.column_stack((y_test, y_hat_test))
    test_y_diff = pd.DataFrame(y_comp, columns=["Y", "Y_hat"])
    print(test_y_diff)
    
