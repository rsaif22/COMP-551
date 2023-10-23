import numpy as np 
from node import *
# Implement the MLP as a python class. The constructor for the class should take as input the activation function
# (e.g., ReLU), the number of hidden layers (e.g., 2) and the number of units in the hidden layers (e.g., [64, 64]) and
# it should initialize the weights and biases (with an
class MLP:
    def __init__(self, activation_func: str, n_hidden_layers: int, hidden_units: list,
                  input_dims: int, output_dims: int, initialization: str):
        self.activation_func = activation_func
        self.n_hidden = n_hidden_layers
        self.hidden_units = hidden_units
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_classes = output_dims
        self.initialization = initialization
        self.weights = []
        self.nodes = []
        # First weight
        self.init_weights()
        self.init_nodes(activation_func = activation_func)

    def init_weights(self):
        if self.initialization == "zeros":
            w_in = np.zeros((self.input_dims + 1, self.hidden_units))
            self.weights.append(w_in)
            # For all layers except last layer, weights between hidden layers
            w = np.zeros((self.hidden_units + 1, self.hidden_units)) # For all hidden layers
            for i in range(self.n_hidden - 1):
                self.weights.append(w)
            # For output
            w_out = np.zeros((self.hidden_units + 1, self.output_dims))
            self.weights.append(w_out)
        if self.initialization == "ones":
            w_in = np.ones((self.input_dims + 1, self.hidden_units))
            self.weights.append(w_in)
            # For all layers except last layer, weights between hidden layers
            w = np.ones((self.hidden_units + 1, self.hidden_units)) # For all hidden layers
            for i in range(self.n_hidden - 1):
                self.weights.append(w)
            # For output
            w_out = np.ones((self.hidden_units + 1, self.output_dims))
            self.weights.append(w_out)
        if self.initialization == "uniform":
            w_in = np.random.uniform(0, 1, (self.input_dims + 1, self.hidden_units))
            self.weights.append(w_in)
            # For all layers except last layer, weights between hidden layers
            w = np.random.uniform(0, 1, (self.hidden_units + 1, self.hidden_units)) # For all hidden layers
            for i in range(self.n_hidden - 1):
                self.weights.append(w)
            # For output
            w_out = np.random.uniform(0, 1, (self.hidden_units + 1, self.output_dims))
            self.weights.append(w_out)

    def init_nodes(self, activation_func: Node):
        # Initialize all weights and activations
        activation = None
        match activation_func:
            case "ReLu":
                activation = ReLu
        for i in range (len(self.weights)-1):
            self.nodes.append(Multiply())
            self.nodes.append(activation())
        self.nodes.append(Multiply())
        self.nodes.append(Softmax())

    def forward(self, X: np.ndarray):
        # Predict
        output = X 
        weight_idx = 0
        for node in self.nodes:
            if type(node) == Multiply:
                output = node.forward(output, self.weights[weight_idx])
                weight_idx += 1
            else:
                output = node.forward(output)
        return output
    
    def compute_gradients(self, true_y: np.ndarray):
        weight_grads = []
        d_out = self.nodes[-1].backward(true_y) # initial output gradient (10 X 10)
        reversed_nodes = self.nodes.copy()
        reversed_nodes.reverse()
        for node in reversed_nodes:
            if type(node) == Multiply:
                d_out, weight_grad = node.backward(d_out)
                weight_grads.append(weight_grad)
            elif type(node) == ReLu:
                d_out = node.backward(d_out)
        weight_grads.reverse() # Flip to get right order
        return weight_grads
    
    def encode_y(self, y: np.ndarray):
        y = np.reshape(y, (y.size, 1))
        encoded_y = np.broadcast_to(y, (y.size, self.num_classes)).copy()
        for i in range(self.num_classes):
            class_num = i # Zero indexed
            encoded_y[:, i] = encoded_y[:, i] == class_num
        return encoded_y
    
    def fit(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, epsilon: float = 1e-4,
            batch_size:int = 10, max_iters:int = 1e3):
        # self.f1_list = np.empty((3, 0))
        # self.time_list = np.empty((0,))
        y_encoded = self.encode_y(y)
        all_indices = np.arange(y_encoded.shape[0])
        current_batch = np.random.choice(all_indices, batch_size)
        # forward pass
        y_hat = self.forward(X[current_batch, :])
        weight_grads = self.compute_gradients(y_encoded[current_batch, :])
        num_iters = 0
        # time_start = time.time()
        while(num_iters < max_iters):
            # self.f1_list = np.column_stack((self.f1_list, self.compute_F1(X, y)))
            # time_from_start = time.time() - time_start
            # self.time_list = np.append(self.time_list, time_from_start)
            # self.w = self.w - learning_rate*grad
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] - learning_rate * weight_grads[i]
            current_batch = np.random.choice(all_indices, batch_size, replace=False)
            y_hat = self.forward(X[current_batch, :])
            weight_grads = self.compute_gradients(y_encoded[current_batch, :])
            num_iters += 1
        print("Finished")

    
if __name__=="__main__":
    mlp = MLP("ReLu", 2, 16, 100, 5)
    input = np.ones((100, 1))

    