import numpy as np 

class Node:
    def __init__(self):
        pass
    
    def forward(self, X: np.ndarray):
        raise NotImplementedError
    
    def backward(self, dz: np.ndarray):
        raise NotImplementedError

# D X 2 features
# weight: D X 1
# output: 2 X 1
# bias: 2 X 1
class Add(Node):
    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray, bias: np.ndarray):
        assert bias.shape[0] == input.shape[0], "Bias dimensions do not match input"
        self.input = input
        self.bias = bias 
        return self.input + self.bias
    
    def backward(self, output_grad: np.ndarray):
        # Inputs: N X D, Bias: N X 1, output_grad: N X (D+1) 
        grad_inputs = output_grad 

class Multiply(Node):
    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray, weights: np.ndarray, add_bias:bool = False):
        if add_bias:
            assert weights.shape[0] == (input.shape[1] + 1), "Input dimensions do not match"
        else:
            assert weights.shape[0] == input.shape[1], "Input dimensions do not match"
        self.input = input
        self.add_bias = add_bias
        if add_bias:
            self.input = np.append(self.input, np.ones((self.input.shape[0], 1)), axis=1)
        self.weights = weights
        N = weights.shape[0]
        return (self.input @ self.weights)
    
    def backward(self, output_grad):
        N = output_grad.shape[0]
        local_grad_inputs = self.weights 
        local_grad_weights = self.input
        try:
            input_grad = output_grad @ local_grad_inputs.T
        except:
            input_grad = np.outer(output_grad, local_grad_inputs)
        if self.add_bias:
            input_grad = np.delete(input_grad, -1, axis=1)
        weight_grad = local_grad_weights.T @ output_grad / N
        return input_grad, weight_grad
    
class ReLU(Node):
    def __init__(self):
        super().__init__()
    
    def forward(self, input: np.ndarray):
        self.input = input # Store for backward pass
        return np.maximum(self.input, np.zeros(self.input.shape))
    
    def backward(self, output_grad: np.ndarray):
        local_grad = np.ones(self.input.shape) # Derivative of x
        local_grad[self.input < 0] = 0
        return local_grad * output_grad
    
class LogisticOutput(Node):
    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray):
        self.input = input
        self.output = 1 / (1 + np.exp(-input))
        return self.output
    
    def backward(self, output_true: np.ndarray):
        return self.output - output_true
    
class Logistic(Node):
    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray):
        self.input = input
        self.output = 1 / (1 + np.exp(-input))
        return self.output
    
    def backward(self, output_grad: np.ndarray):
        local_grad = self.output * (1 - self.output)
        return output_grad * local_grad

    
class Softmax(Node):
    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray):
        self.input = input 
        input_shifted = input.copy()
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
    # X = np.random.uniform(0, 1, (10, 5))
    # print(X)
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


    # ###########
    # output_true = np.ones(out4.shape)
    # d_out4 = smax.backward(output_true)
    # d_out3, d_w2 = mul2.backward(d_out4)
    # d_out2 = relu.backward(d_out3)
    # d_out1, d_w1 = mul1.backward(d_out2)
    # print(d_out1.shape)

    # X = np.array(([0.1, 0.2], [0.3, 0.4], [0.5, 0.6]))
    # w1 = np.array(([0.1, 0.2], [0.3, 0.4], [0, 0]))
    # mul1 = Multiply()
    # print(mul1.forward(X, w1))
    logistic = lambda z: 1./ (1 + np.exp(-z))
    x = np.random.randn(4, 3)
    N,D = x.shape
    v = np.random.randn(3, 5)
    w = np.random.randn(5)
    z = logistic(np.dot(x, v)) #N x M
    log1 = Logistic()
    log2 = LogisticOutput()
    mul1 = Multiply()
    mul2 = Multiply()
    z_m = log1.forward(mul1.forward(x, v, False))
    yh = logistic(np.dot(z, w))#N
    yh_m = log2.forward(mul2.forward(z, w, False))
    y = np.array([0, 1, 0, 1])
    # print(yh)
    # print("--------------")
    # print(yh_m)
    dy = yh - y #N
    dy_m = log2.backward(y)
    dw = np.dot(z.T, dy)/N #M
    dz_m, dw_m = mul2.backward(dy_m)
    dz = np.outer(dy, w) #N x M
    dv = np.dot(x.T, dz * z * (1 - z))/N #D x M
    dlog = dz * z * (1 - z)
    dlog_m = log1.backward(dz_m)
    dx_m, dv_m = mul1.backward(dlog_m)
    print(dv)
    print("--------------")
    print(dv_m)