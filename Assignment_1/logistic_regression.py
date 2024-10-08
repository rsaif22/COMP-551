import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, D: int, num_classes: int):
        self.w = np.zeros((D+1, num_classes)) # column for each class
        self.mean = np.zeros((D,))
        self.std = np.zeros((D,))
        self.num_classes = num_classes
        self.f1_list = np.empty((3, 1))
    
    def softmax(self, z:np.ndarray):
        softmax_matrix = np.zeros(z.shape)
        for i in range(z.shape[1]):
            softmax_matrix[:, i] = np.exp(z[:, i]) / np.sum(np.exp(z), axis=1)
        return softmax_matrix
    
    def compute_F1(self, X: np.ndarray, y: np.ndarray):
        y_encoded = self.encode_y(y)
        y_hat_encoded = self.encode_y(self.predict(X))
        J = np.zeros((y_encoded.shape[1], 1))
        for i in range(y_encoded.shape[1]):
            y_column = y_encoded[:, i]
            y_hat_column = y_hat_encoded[:, i]
            y_true = y_column == 1
            y_hat_true = y_hat_column==1
            true_positives = np.sum((y_true==1) & (y_hat_true==1))
            false_positives = np.sum((y_true==0) & (y_hat_true==1))
            false_negatives = np.sum((y_true==1) & (y_hat_true==0))
            epsilon = 1e-8
            precision = float(true_positives) / float(true_positives + false_positives + epsilon)
            recall = float(true_positives) / float(true_positives + false_negatives + epsilon)
            F1 = (2 * precision * recall) / (precision + recall + epsilon)
            J[i] = F1
        return J

    def compute_gradient(self, X: np.ndarray, y: np.ndarray):
        N = y.shape[0]
        y_hat = self.softmax(X @ self.w)
        return 1/N * X.T @ (y_hat-y)
    
    def encode_y(self, y):
        y = np.reshape(y, (y.size, 1))
        encoded_y = np.broadcast_to(y, (y.size, self.num_classes)).copy()
        for i in range(self.num_classes):
            class_num = i + 1 # Not zero indexed
            encoded_y[:, i] = encoded_y[:, i] == class_num
        return encoded_y
            
    def fit(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, epsilon: float = 1e-4,
            max_iters:int = 1e3):
        self.w = np.zeros(self.w.shape) # column for each class
        self.f1_list = np.empty((3, 0))
        self.mean = np.mean(X, axis=0)
        self.std = np.mean(X, axis=0)
        X_normalized = self.normalize(X)
        y_encoded = self.encode_y(y)
        grad = self.compute_gradient(X_normalized, y_encoded)
        num_iters = 0
        while(np.linalg.norm(grad) > epsilon and num_iters < max_iters):
            #print(self.compute_F1(X, y)[1])
            self.f1_list = np.column_stack((self.f1_list, self.compute_F1(X, y)))
            self.w = self.w - learning_rate*grad
            grad = self.compute_gradient(X_normalized, y_encoded)
            num_iters += 1
    
    def predict(self, X):
        X_normalized = self.normalize(X)
        y_probs =  self.softmax(X_normalized @ self.w)
        y_hat = np.argmax(y_probs, axis=1) + 1
        return y_hat
    
    def normalize(self, X: np.ndarray, add_bias: bool = True):
        X_normalized = (X - self.mean) / self.std
        if add_bias:
            X_normalized = np.column_stack((X_normalized, np.ones((X_normalized.shape[0], 1))))
        return X_normalized


if __name__=="__main__":
    from ucimlrepo import fetch_ucirepo 
  
    # fetch dataset 
    wine = fetch_ucirepo(id=109) 
    
    # data (as pandas dataframes) 
    X = wine.data.features.to_numpy()
    y = wine.data.targets.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    logistic_reg = LogisticRegression(X.shape[1], 3)
    logistic_reg.fit(X_train, y_train)

    y_hat_test = logistic_reg.predict(X_test)
    y_comp = np.column_stack((y_test, y_hat_test))
    test_y_diff = pd.DataFrame(y_comp, columns=["Y", "Y_hat"])
    print(logistic_reg.compute_F1(X_test, y_test))
    
