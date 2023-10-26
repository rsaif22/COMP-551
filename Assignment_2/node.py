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

    def forward(self, input: np.ndarray, weights: np.ndarray):
        assert weights.shape[0] == input.shape[1], "Input dimensions do not match"
        self.input = input
        self.weights = weights
        return (self.input @ self.weights)
    
    def backward(self, output_grad, regularization: str = None, lambda_: float = 0.01):
        N = output_grad.shape[0]
        local_grad_inputs = self.weights 
        local_grad_weights = self.input
        try:
            input_grad = output_grad @ local_grad_inputs.T
        except:
            input_grad = np.outer(output_grad, local_grad_inputs)
        weight_grad = local_grad_weights.T @ output_grad / N
        if regularization == "l2":
            weight_grad += lambda_ * self.weights
        elif regularization == "l1":
            weight_grad += lambda_ * np.sign(self.weights)
        return input_grad, weight_grad
    
class Add:
    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray, bias: np.ndarray):
        # Input is N X D dim, bias should be 1 X D dim
        self.input = input
        bias = bias.reshape((1, bias.size))
        self.bias = bias
        bias = np.broadcast_to(bias, input.shape)
        return input + bias
    
    def backward(self, output_grad: np.ndarray, regularization: str = None, lambda_: float = 0.01):
        # output grad should be N X D dimensional
        local_grad_input = np.ones(output_grad.shape) # N X D
        local_grad_bias = np.ones(output_grad.shape) # 1 X D
        N = output_grad.shape[0]
        grad_input = local_grad_input * output_grad
        grad_bias = np.mean(local_grad_bias * output_grad, axis=0)
        if regularization == "l2":
            grad_bias += lambda_ * self.bias.reshape((grad_bias.shape))
        elif regularization == "l1":
            grad_bias += lambda_ * np.sign(self.bias.reshape((grad_bias.shape)))
        return grad_input, grad_bias
    
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
    
class LeakyReLU(Node):
    def __init__(self):
        super().__init__()
    
    def forward(self, input: np.ndarray, gamma: float = 0.05):
        self.input = input # Store for backward pass
        self.gamma = gamma
        output = np.maximum(self.input, np.zeros(self.input.shape)) + \
                gamma * np.minimum(self.input, np.zeros(self.input.shape))
        return output
    
    def backward(self, output_grad: np.ndarray):
        local_grad = np.ones(self.input.shape) # Derivative of x
        local_grad[self.input < 0] = self.gamma
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
    
class Tanh(Node):
    def __init__(self):
        super().__init__()
    
    def forward(self, input: np.ndarray):
        self.input = input
        self.output = np.tanh(input)
        return self.output
    
    def backward(self, output_grad: np.ndarray):
        local_grad = 1 - self.output ** 2
        return output_grad * local_grad

class Softmax(Node):
    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray, epsilon: float = 1e-8):
        self.input = input 
        input_shifted = input.copy()
        max_values = np.max(input, 1).reshape((input.shape[0], 1))
        input_shifted -= max_values
        input_exp = np.exp(input_shifted)
        exp_sum = np.sum(input_exp, axis=1).reshape((input_exp.shape[0], 1)) + epsilon
        self.output = input_exp / exp_sum # Store this as we will use this in gradient
        return self.output 
    
    def backward(self, output_true: np.ndarray): # As this is final layer, we only need the true y values
        return self.output - output_true # Gives derivative of loss directly
    

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
    bias1 = np.random.randn(1, 5)
    bias2 = np.random.randn(1, 1)
    w = np.random.randn(5)
    z = logistic(np.dot(x, v)) #N x M
    log1 = Logistic()
    log2 = LogisticOutput()
    mul1 = Multiply()
    add1 = Add()
    mul2 = Multiply()
    add2 = Add()
    # z1 = mul1.forward(x, v, False)
    # added = add1.forward(z1, bias1)
    # print(added)
    m1 = mul1.forward(x, v, False)
    b1 = add1.forward(m1, bias1)
    z_m = log1.forward(b1)
    yh = logistic(np.dot(z, w))#N
    m2 = mul2.forward(z, w, False)
    #b2 = add2.forward(m2, bias2)
    yh_m = log2.forward(m2)
    y = np.array([0, 1, 0, 1])
    print(yh)
    print("--------------")
    print(yh_m)
    dy = yh - y #N
    dy_m = log2.backward(y)
    #db1_m = add2.backward(dy_m)
    dw = np.dot(z.T, dy)/N #M
    dz_m, dw_m = mul2.backward(dy_m)
    dz = np.outer(dy, w) #N x M
    dv = np.dot(x.T, dz * z * (1 - z))/N #D x M
    dlog = dz * z * (1 - z)
    dlog_m = log1.backward(dz_m)
    dt = add1.backward(dlog_m)
    dx_m, dv_m = mul1.backward(dlog_m)
    print(dv)
    print("--------------")
    print(dv_m)