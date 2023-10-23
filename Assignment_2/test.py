# Import necessary libraries
import numpy as np
from node import *

# Define your random data
np.random.seed(0)
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = np.random.randint(0, 2, size=(100, 2))  # Binary labels

# Define a simple network structure (5 input features, 3 hidden units, 1 output unit)
# with a ReLU activation function in the hidden layer
input_size = X.shape[1]
hidden_units = 3
output_units = 2

# Initialize weights and biases randomly
weights_hidden = np.random.rand(input_size + 1, hidden_units)
weights_output = np.random.rand(hidden_units + 1, output_units)

# Create instances of the nodes
multiply_1 = Multiply()
multiply_2 = Multiply()
relu_node = ReLu()
softmax_node = Softmax()

# Forward pass
hidden_layer_input = multiply_1.forward(X, weights_hidden)
hidden_layer_output = relu_node.forward(hidden_layer_input)
output = multiply_2.forward(hidden_layer_output, weights_output, add_bias=False)  # No bias for output layer

# Apply softmax to get probabilities
probabilities = softmax_node.forward(output)

# Print some results for verification
print("Input:")
print(X)

print("\nHidden Layer Input:")
print(hidden_layer_input)

print("\nHidden Layer Output:")
print(hidden_layer_output)

print("\nOutput:")
print(output)

print("\nProbabilities:")
print(probabilities)
