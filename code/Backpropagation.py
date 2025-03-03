import numpy as np

class NeuralNetwork:
    def __init__(self):
        # Initialize inputs
        self.i1 = 0.05
        self.i2 = 0.10
        
        # Initialize weights
        self.w1 = 0.15
        self.w2 = 0.20
        self.w3 = 0.25
        self.w4 = 0.30
        self.w5 = 0.40
        self.w6 = 0.45
        self.w7 = 0.50
        self.w8 = 0.55
        
        # Initialize biases
        self.b1 = 0.35
        self.b2 = 0.60
        
        # Initialize targets
        self.t1 = 0.01
        self.t2 = 0.99
        
        # Learning rate
        self.learning_rate = 0.5

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self):
        # Hidden layer calculations
        self.h1_in = self.w1 * self.i1 + self.w2 * self.i2 + self.b1
        self.h1_out = self.sigmoid(self.h1_in)
        
        self.h2_in = self.w3 * self.i1 + self.w4 * self.i2 + self.b1
        self.h2_out = self.sigmoid(self.h2_in)
        
        # Output layer calculations
        self.o1_in = self.w5 * self.h1_out + self.w6 * self.h2_out + self.b2
        self.o1_out = self.sigmoid(self.o1_in)
        
        self.o2_in = self.w7 * self.h1_out + self.w8 * self.h2_out + self.b2
        self.o2_out = self.sigmoid(self.o2_in)
        
        # Calculate total error
        self.error = 0.5 * ((self.t1 - self.o1_out)**2 + (self.t2 - self.o2_out)**2)
        
        return self.o1_out, self.o2_out, self.error

    def backward_propagation(self):
        # Output layer deltas
        self.delta_o1 = (self.o1_out - self.t1) * self.sigmoid_derivative(self.o1_out)
        self.delta_o2 = (self.o2_out - self.t2) * self.sigmoid_derivative(self.o2_out)
        
        # Hidden layer deltas
        self.delta_h1 = (self.delta_o1 * self.w5 + self.delta_o2 * self.w7) * self.sigmoid_derivative(self.h1_out)
        self.delta_h2 = (self.delta_o1 * self.w6 + self.delta_o2 * self.w8) * self.sigmoid_derivative(self.h2_out)
        
        # Update weights and biases
        # Output layer weights
        self.w5 -= self.learning_rate * self.delta_o1 * self.h1_out
        self.w6 -= self.learning_rate * self.delta_o1 * self.h2_out
        self.w7 -= self.learning_rate * self.delta_o2 * self.h1_out
        self.w8 -= self.learning_rate * self.delta_o2 * self.h2_out
        
        # Hidden layer weights
        self.w1 -= self.learning_rate * self.delta_h1 * self.i1
        self.w2 -= self.learning_rate * self.delta_h1 * self.i2
        self.w3 -= self.learning_rate * self.delta_h2 * self.i1
        self.w4 -= self.learning_rate * self.delta_h2 * self.i2
        
        # Update biases
        self.b2 -= self.learning_rate * (self.delta_o1 + self.delta_o2)
        self.b1 -= self.learning_rate * (self.delta_h1 + self.delta_h2)

# Create and train the network
nn = NeuralNetwork()

print("Initial weights:")
print(f"w1-w4: {nn.w1:.4f}, {nn.w2:.4f}, {nn.w3:.4f}, {nn.w4:.4f}")
print(f"w5-w8: {nn.w5:.4f}, {nn.w6:.4f}, {nn.w7:.4f}, {nn.w8:.4f}")

# Initial forward pass
outputs = nn.forward_propagation()
print("\nInitial outputs:")
print(f"Outputs (o1, o2): {outputs[0]:.4f}, {outputs[1]:.4f}")
print(f"Initial Error: {outputs[2]:.4f}")

# Perform backward propagation
nn.backward_propagation()

# Forward pass with updated weights
new_outputs = nn.forward_propagation()
print("\nAfter one training step:")
print(f"Updated weights:")
print(f"w1-w4: {nn.w1:.4f}, {nn.w2:.4f}, {nn.w3:.4f}, {nn.w4:.4f}")
print(f"w5-w8: {nn.w5:.4f}, {nn.w6:.4f}, {nn.w7:.4f}, {nn.w8:.4f}")
print(f"New outputs (o1, o2): {new_outputs[0]:.4f}, {new_outputs[1]:.4f}")
print(f"New Error: {new_outputs[2]:.4f}")
