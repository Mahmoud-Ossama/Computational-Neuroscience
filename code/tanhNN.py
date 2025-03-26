import numpy as np

np.random.seed(42)

w1 = np.random.uniform(-0.5, 0.5) 
w2 = np.random.uniform(-0.5, 0.5) 

b1 = 0.5 
b2 = 0.7 

x = np.array([0.1, 0.05])

def tanh(x):
    return np.tanh(x)

# forwrd function
def forward(x, w1, b1, w2, b2):

    z1 = np.dot(x, w1) + b1
    a1 = tanh(z1)
    

    z2 = np.dot(a1, w2) + b2
    a2 = tanh(z2)
    
    return a1, a2

hidden_output, final_output = forward(x, w1, b1, w2, b2)

print("Input:", x)
print("\nWeights (hidden layer):\n", w1)
print("\nWeights (output layer):\n", w2)
print("\nBias (hidden layer):", b1)
print("\nBias (output layer):", b2)
print("\nHidden layer output:", hidden_output)
print("\nFinal output:", final_output[0])