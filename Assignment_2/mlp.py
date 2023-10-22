import numpy as np 
from node import *
# Implement the MLP as a python class. The constructor for the class should take as input the activation function
# (e.g., ReLU), the number of hidden layers (e.g., 2) and the number of units in the hidden layers (e.g., [64, 64]) and
# it should initialize the weights and biases (with an
class MLP:
    def __init__(self, activation_func: str, n_hidden_layers: int, hidden_units: list,
                  input_dims: int, output_dims: int):
        self.activation_func = activation_func
        self.n_hidden = n_hidden_layers
        self.hidden_units = hidden_units
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_classes = output_dims
        self.weights = []
        self.nodes = []
        # First weight
        self.init_weights(initiailization="zeros")
        self.init_nodes(activation_func = activation_func)

    def init_weights(self, initiailization: str = "zeros"):
        if initiailization == "zeros":
            w_in = np.zeros((self.input_dims + 1, self.hidden_units))
            self.weights.append(w_in)
            # For all layers except last layer, weights between hidden layers
            w = np.zeros((self.hidden_units + 1, self.hidden_units)) # For all hidden layers
            for i in range(self.n_hidden - 1):
                self.weights.append(w)
            # For output
            w_out = np.zeros((self.hidden_units + 1, self.output_dims))
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
        d_out = self.nodes[-1].backward(true_y)
        # for node in self.nodes:
        #     if type(node) == Multiply:

    def encode_y(self, y: np.ndarray):
        y = np.reshape(y, (y.size, 1))
        encoded_y = np.broadcast_to(y, (y.size, self.num_classes)).copy()
        for i in range(self.num_classes):
            class_num = i # Zero indexed
            encoded_y[:, i] = encoded_y[:, i] == class_num
        return encoded_y

    
if __name__=="__main__":
    mlp = MLP("ReLu", 2, 64, 748, 9)
    # for weights in mlp.weights:
    #     print(weights.shape)
    for node in mlp.nodes:
        print(type(node))
    