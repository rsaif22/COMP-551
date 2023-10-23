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
        N = weights.shape[0]
        return (self.input @ self.weights) / N
    
    def backward(self, output_grad):
        local_grad_inputs = self.weights 
        local_grad_weights = self.input
        input_grad = output_grad @ local_grad_inputs.T
        input_grad = np.delete(input_grad, -1, axis=1)
        weight_grad = local_grad_weights.T @ output_grad
        return input_grad, weight_grad
    
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
        input_shifted = input.copy()
        # for i in range(input.shape[1]):
        #     input_shifted[:, i] = input[:, i] - np.max(input, 1)
        max_values = np.max(input, 1).reshape((input.shape[0], 1))
        input_shifted -= max_values
        input_exp = np.exp(input_shifted)
        exp_sum = np.sum(input_exp, axis=1).reshape((input_exp.shape[0], 1))
        self.output = input_exp / exp_sum # Store this as we will use this in gradient
        return self.output 
    
    def backward(self, output_true: np.ndarray): # As this is final layer, we only need the true y values
        N = np.size(output_true) # Size of output
        return 1/N * self.input * (self.output - output_true) # Gives derivative of loss directly
    

if __name__ == "__main__":
    X = np.random.uniform(0, 1, (2, 2))
    print(X)
    # w1 = np.random.uniform(0 ,1, (6, 16))
    # mul1 = Multiply()
    # relu = ReLu()
    # mul2 = Multiply()
    # smax = Softmax()

    # out1 = mul1.forward(X, w1)
    # out2 = relu.forward(out1)
    # w2 = np.random.uniform(0 ,1, (17, 5))
    # out3 = mul2.forward(out2, w2)  
    # out4 = smax.forward(out3)
    # print(out4[1])


    ###########
    output_true = np.ones(out4.shape)
    d_out4 = smax.backward(output_true)
    d_out3, d_w2 = mul2.backward(d_out4)
    d_out2 = relu.backward(d_out3)
    d_out1, d_w1 = mul1.backward(d_out2)
    # print(d_out1.shape)