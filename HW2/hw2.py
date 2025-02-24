import numpy as np

# ----------------------------
# 1. Training data for XOR
# ----------------------------
# XOR truth table:
# X1  X2 | Y
# 0   0  | 0
# 0   1  | 1
# 1   0  | 1
# 1   1  | 0

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0],
              [1],
              [1],
              [0]])

# ----------------------------
# 2. Hyperparameters
# ----------------------------
input_neurons = 2    # We have 2 inputs
hidden_neurons = 2   # Let's pick 2 neurons in the hidden layer
output_neurons = 1   # We have 1 output (0 or 1)
learning_rate = 0.5
epochs = 20000

# ----------------------------
# 3. Initialize weights/biases
# ----------------------------
# Using small random values for weights
np.random.seed(42)  # For reproducibility

W1 = np.random.randn(input_neurons, hidden_neurons)
b1 = np.random.randn(1, hidden_neurons)
W2 = np.random.randn(hidden_neurons, output_neurons)
b2 = np.random.randn(1, output_neurons)

# ----------------------------
# 4. Activation functions
# ----------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)  # derivative of sigmoid if x = sigmoid(z)

# ----------------------------
# 5. Training loop
# ----------------------------
for i in range(epochs):
    # Forward pass
    # Hidden layer
    z1 = np.dot(X, W1) + b1        # shape: (4, hidden_neurons)
    a1 = sigmoid(z1)               # shape: (4, hidden_neurons)
    
    # Output layer
    z2 = np.dot(a1, W2) + b2       # shape: (4, 1)
    a2 = sigmoid(z2)               # shape: (4, 1) -> predictions

    # Mean Squared Error (MSE) loss
    loss = np.mean((y - a2) ** 2)

    # Backpropagation
    # dL/da2 = (a2 - y), shape: (4, 1)
    d_a2 = (a2 - y)
    # dL/dz2 = dL/da2 * derivative of sigmoid(z2)
    d_z2 = d_a2 * sigmoid_derivative(a2)         # shape: (4, 1)
    
    # Gradients for W2, b2
    dW2 = np.dot(a1.T, d_z2)                     # shape: (hidden_neurons, 1)
    db2 = np.sum(d_z2, axis=0, keepdims=True)    # shape: (1, 1)

    # Backprop through hidden layer
    d_a1 = np.dot(d_z2, W2.T)                    # shape: (4, hidden_neurons)
    d_z1 = d_a1 * sigmoid_derivative(a1)         # shape: (4, hidden_neurons)
    
    # Gradients for W1, b1
    dW1 = np.dot(X.T, d_z1)                      # shape: (2, hidden_neurons)
    db1 = np.sum(d_z1, axis=0, keepdims=True)    # shape: (1, hidden_neurons)
    
    # Update weights and biases
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # (Optional) print the loss every 1000 epochs
    if (i+1) % 1000 == 0:
        print(f"Epoch {i+1}/{epochs}, Loss: {loss:.6f}")

# ----------------------------
# 6. Testing the final network
# ----------------------------
print("\nFinal predictions after training:")
for idx, x_val in enumerate(X):
    # Forward pass with final weights
    hidden_out = sigmoid(np.dot(x_val, W1) + b1)
    output = sigmoid(np.dot(hidden_out, W2) + b2)
    print(f"Input: {x_val}, Predicted: {output.item():.4f}, Target: {y[idx, 0]}")
