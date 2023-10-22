import numpy as np 

class Node:
    def __init__(self):
        pass
    
    def forward(self, X: np.ndarray):
        raise NotImplementedError
    
    def backward(self, dz: np.ndarray):
        raise NotImplementedError
    
class Multiply(Node):
    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray, weights: np.ndarray, add_bias:bool = True):
        assert weights.shape[0] == (input.shape[1] + 1), "Input dimensions do not match"
        self.input = input
        if add_bias:
            self.input = np.append(self.input, np.ones((self.input.shape[0], 1)), axis=1)
        self.weights = weights
        return self.input @ self.weights
    
    def backward(self, output_grad):
        local_grad_inputs = self.weights 
        local_grad_weights = self.input
        return local_grad_inputs @ output_grad, local_grad_weights @ output_grad.T
    
class ReLu(Node):
    def __init__(self):
        super().__init__()
    
    def forward(self, input: np.ndarray):
        self.input = input # Store for backward pass
        return np.maximum(self.input, np.zeros(self.input.shape))
    
    def backward(self, output_grad: np.ndarray):
        local_grad = np.ones(self.input.shape) # Derivative of x
        local_grad[self.input < 0] = 0
        return local_grad * output_grad
    
class Softmax(Node):
    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray):
        self.input = input 
        input_exp = np.exp(input)
        exp_sum = np.sum(input_exp)
        self.output = input_exp / exp_sum # Store this as we will use this in gradient
        return self.output 
    
    def backward(self, output_true: np.ndarray): # As this is final layer, we only need the true y values
        N = np.size(output_true) # Size of output
        return 1/N * self.input.T @ (self.output - output_true) # Gives derivative of loss directly
    

if __name__ == "__main__":
    X = np.array(([1, -2], [-3, 4], [5, 6]))
    # print(X)
    # w = np.array(([-1], [-2], [-3]))
    # node1 = Multiply()
    # print(node1.forward(X, w))
    # dz = np.array([0, 1]).reshape((2, 1))
    # print(node1.backward(dz))
    node1 = Softmax()
    print(node1.forward(X))
    #true_y = np.array([0.09, 0.2447, 1]).reshape((3, 1))
    # print(node1.backward(true_y))

    # node1 = ReLu()
    # print(node1.forward(X))
    # dz = np.ones(X.shape)
    # print(node1.backward(dz))