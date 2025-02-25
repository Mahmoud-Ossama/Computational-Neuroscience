import numpy as np

def tanh(x):
    return np.tanh(x)

np.random.seed(42)

w1 = np.random.uniform(-0.5, 0.5, (2, 2))
w2 = np.random.uniform(-0.5, 0.5, (1, 2))

b1 = 0.5
b2 = 0.7

x = np.array([[0.1, 0.2]])

z1 = np.dot(x, w1) + b1
a1 = tanh(z1)

z2 = np.dot(a1, w2.T) + b2
output = tanh(z2)

print("Weights (w1):\n", w1)
print("Weights (w2):\n", w2)
print("Hidden layer output:\n", a1)
print("Network output:\n", output)
