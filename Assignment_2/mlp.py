import numpy as np 
from node import *
# Implement the MLP as a python class. The constructor for the class should take as input the activation function
# (e.g., ReLU), the number of hidden layers (e.g., 2) and the number of units in the hidden layers (e.g., [64, 64]) and
# it should initialize the weights and biases (with an
class MLP:
    def __init__(self, activation_func: str, n_hidden_layers: int, hidden_units: list,
                  input_dims: int, output_dims: int, initialization: str, seed: int = None):
        self.activation_func = activation_func
        self.n_hidden = n_hidden_layers
        self.hidden_units = hidden_units
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_classes = output_dims
        self.init_func = None
        self.weights = []
        self.biases = []
        self.nodes = []
        self.acc_list = None # Accuracies for plotting
        self.weight_norms = None # Weight norms for plotting
        if seed is not None:
            np.random.seed(seed)
        self.select_init_func(initialization)
        self.init_weights()
        self.init_biases()
        self.init_nodes()

    def select_init_func(self, init_func: str):
        match init_func:
            case "zeros":
                self.init_func = self.zeros
            case "uniform":
                self.init_func = self.uniform 
            case "randn":
                self.init_func = self.randn
            case "gaussian":
                self.init_func = self.gaussian
            case "xavier":
                self.init_func = self.xavier
            case "kaiming":
                self.init_func = self.kaiming
    
    def init_weights(self):
        # First layer
        if self.n_hidden == 0: 
            w = self.init_func(self.input_dims, self.output_dims)
            self.weights.append(w) # Only one weight in this case
        else:
            w_in = self.init_func(self.input_dims, self.hidden_units)
            self.weights.append(w_in)
            for i in range(self.n_hidden - 1):
                w = self.init_func(self.hidden_units, self.hidden_units)
                self.weights.append(w)
            w_out = self.init_func(self.hidden_units, self.output_dims)
            self.weights.append(w_out)

    def init_biases(self):
        for weight in self.weights:
            current_bias = self.init_func(1, weight.shape[1])
            self.biases.append(current_bias)

    def init_nodes(self):
        activation_func = None 
        match self.activation_func:
            case "ReLU":
                activation_func = ReLU
            case "logistic":
                activation_func = Logistic
            case "LeakyReLU":
                activation_func = LeakyReLU
            case "tanh":
                activation_func = Tanh
        for i in range(len(self.weights)-1):
            self.nodes.append(Multiply())
            self.nodes.append(Add())
            self.nodes.append(activation_func())
        self.nodes.append(Multiply())
        self.nodes.append(Add())
        if self.output_dims == 1:
            self.nodes.append(LogisticOutput())
        else:
            self.nodes.append(Softmax())

        
    def zeros(self, rows, cols):
        return np.zeros((rows, cols))
    
    def ones(self, rows, cols):
        return np.ones((rows, cols))
    
    def uniform(self, rows, cols):
        return np.random.uniform(-1, 1, (rows, cols))
    
    def randn(self, rows, cols):
        return np.random.randn(rows, cols) * 0.01 # Based on tutorial
    
    def gaussian(self, rows, cols):
        return np.random.normal(0, 1, (rows, cols))

    def xavier(self, rows, cols):
        std = np.sqrt(2 / (rows + cols))
        return np.random.normal(0, std, (rows, cols))
    
    def kaiming(self, rows, cols):
        std = np.sqrt(2 / rows)
        return np.random.normal(0, std, (rows, cols))
        
    def forward(self, X: np.ndarray):
        input = X.copy() # Start with initial input
        weight_idx = 0
        bias_idx = 0
        for node in self.nodes:
            if type(node) == Multiply:
                input = node.forward(input, self.weights[weight_idx])
                weight_idx += 1
            elif type(node) == Add:
                input = node.forward(input, self.biases[bias_idx])
                bias_idx += 1
            else:
                input = node.forward(input)
        return input

    def backward(self, y: np.ndarray, regularization: str = None, lambda_: float = 0.01):
        if self.output_dims == 1:
            y = y.reshape(y.size, 1)
        node_idx = len(self.nodes) - 1 # Start from last element
        weight_grads = []
        bias_grads = []
        output_grad = y 
        for i in range(node_idx, -1, -1):
            node = self.nodes[i]
            if type(node) == Multiply:
                output_grad, w_grad = node.backward(output_grad, regularization, lambda_)
                weight_grads.append(w_grad)
            elif type(node) == Add:
                output_grad, b_grad = node.backward(output_grad, regularization, lambda_)
                bias_grads.append(b_grad)
            else:
                output_grad = node.backward(output_grad)
        weight_grads.reverse()
        bias_grads.reverse()
        return weight_grads, bias_grads
    
    def encode_y(self, y: np.ndarray):
        y = np.reshape(y, (y.size, 1))
        encoded_y = np.broadcast_to(y, (y.size, self.num_classes)).copy()
        for i in range(self.num_classes):
            class_num = i # Zero indexed
            encoded_y[:, i] = encoded_y[:, i] == class_num
        return encoded_y
    
    def decode_y(self, y_encoded: np.ndarray):
        y_decoded = np.argmax(y_encoded, axis=1).reshape((y_encoded.shape[0], 1))
        return y_decoded
    
    def fit(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.1, epsilon: float = 1e-8, max_iters:int = 1e4, batch_size:int = 10, regularization: str = None, lambda_: float = 0.1):
        self.acc_list = np.empty((0,))
        self.weight_norms = np.empty((0,))
        y_encoded = self.encode_y(y)
        all_indices = np.arange(y_encoded.shape[0])
        current_batch = np.random.choice(all_indices, batch_size)
        # forward pass
        y_hat = self.forward(X[current_batch, :])
        accuracy = self.evaluate_acc(y[current_batch], self.decode_y(y_hat))
        self.acc_list = np.append(self.acc_list, accuracy)
        weight_grads, bias_grads = self.backward(y_encoded[current_batch, :], regularization, lambda_)
        num_iters = 0
        max_norm = np.inf
        while(num_iters < max_iters and max_norm > epsilon):
            max_norm = 0
            weight_norm = 0
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] - learning_rate * weight_grads[i]
                self.biases[i] = self.biases[i] - learning_rate * bias_grads[i]
                if np.linalg.norm(weight_grads[i]) > max_norm:
                    max_norm = np.linalg.norm(weight_grads[i])
                if np.linalg.norm(bias_grads[i]) > max_norm:
                    max_norm = np.linalg.norm(bias_grads[i])
                weight_norm += np.linalg.norm(self.weights[i])
                weight_norm += np.linalg.norm(self.biases[i])
            self.weight_norms = np.append(self.weight_norms, weight_norm)
            y_hat = self.forward(X[current_batch, :])
            accuracy = self.evaluate_acc(y[current_batch], self.decode_y(y_hat))
            self.acc_list = np.append(self.acc_list, accuracy)
            weight_grads, bias_grads = self.backward(y_encoded[current_batch, :], regularization, lambda_)
            current_batch = np.random.choice(all_indices, batch_size, replace=False)
            num_iters += 1
        print(f"Finished in {num_iters}")
    
    def predict(self, X: np.ndarray):
        yh_encoded = self.forward(X)
        return self.decode_y(yh_encoded)
    
    def evaluate_acc(self, y, y_hat):
        y = y.reshape((y.size, 1))
        correct_predictions = float(np.sum(y == y_hat))
        total = float(y.size)
        return correct_predictions / total


from sklearn import datasets
import numpy as np
dataset = datasets.load_iris()
x, y = dataset['data'][:,[1,2]], dataset['target']
y =  y == 1

if __name__=="__main__":
    mlp = MLP("logistic", 1, 32, 2, 1, "randn", 10)
    #print(x.shape)
    # for node in mlp.nodes:
    #     print(type(node))
    print(mlp.weights[0].shape)
    print(mlp.biases[0].shape)
    # for node in mlp.nodes_reversed:
    #     print(type(node))
    #mlp.forward(x)
    #mlp.fit(x, y, learning_rate=0.1, max_iters=20000)
    # print(mlp.backward(y)[1])
    #print(mlp.forward(x))
    # y_encoded = np.array(([0, 1, 0], [1, 0, 0.3], [0, 0.4, 0.7]))
    # print(y_encoded)
    # y = mlp.decode_y(y_encoded)
    # print(y)