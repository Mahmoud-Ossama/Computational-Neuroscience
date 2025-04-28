"""
Build an RNN for a text contains 4 words. The target is predicting the 4th word.
"""

import numpy as np

vocabulary = ['Mahmoud_Osama', 'is', 'the', 'best']
vocab_size = len(vocabulary)
word_to_idx = {word: i for i, word in enumerate(vocabulary)}
idx_to_word = {i: word for i, word in enumerate(vocabulary)}

def one_hot_encode(word):
    one_hot = np.zeros(vocab_size)
    one_hot[word_to_idx[word]] = 1
    return one_hot

input_sequence = ['Mahmoud_Osama', 'is', 'the']
target_word = 'best'

X = [one_hot_encode(word) for word in input_sequence]
y = one_hot_encode(target_word)

hidden_size = 5
learning_rate = 0.01
epochs = 1000

np.random.seed(42)
Wx = np.random.randn(hidden_size, vocab_size) * 0.01
Wh = np.random.randn(hidden_size, hidden_size) * 0.01
b = np.zeros((hidden_size, 1))
Wy = np.random.randn(vocab_size, hidden_size) * 0.01
by = np.zeros((vocab_size, 1))

def forward_pass(X):
    h = np.zeros((hidden_size, 1))
    h_states = [h]
    y_preds = []
    
    for x_t in X:
        x_t = x_t.reshape(-1, 1)
        h = np.tanh(np.dot(Wx, x_t) + np.dot(Wh, h) + b)
        h_states.append(h)
        
        z = np.dot(Wy, h) + by
        y_pred = np.exp(z) / np.sum(np.exp(z))
        y_preds.append(y_pred)
    
    return y_preds, h_states

def compute_loss(y_pred, y_true):
    y_true = y_true.reshape(-1, 1)
    loss = -np.sum(y_true * np.log(y_pred + 1e-8))
    return loss

def backward_pass(X, y, y_preds, h_states):
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)
    dWy = np.zeros_like(Wy)
    dby = np.zeros_like(by)
    
    y_pred = y_preds[-1]
    y_true = y.reshape(-1, 1)
    
    dy = y_pred - y_true
    dWy = np.dot(dy, h_states[-1].T)
    dby = dy
    
    dh_next = np.dot(Wy.T, dy)
    
    for t in reversed(range(len(X))):
        h_t = h_states[t+1]
        h_prev = h_states[t]
        
        dtanh = (1 - h_t * h_t) * dh_next
        
        db += dtanh
        dWx += np.dot(dtanh, X[t].reshape(1, -1))
        dWh += np.dot(dtanh, h_prev.T)
        
        dh_next = np.dot(Wh.T, dtanh)
    
    return dWx, dWh, db, dWy, dby

for epoch in range(epochs):
    y_preds, h_states = forward_pass(X)
    
    loss = compute_loss(y_preds[-1], y)
    
    dWx, dWh, db, dWy, dby = backward_pass(X, y, y_preds, h_states)
    
    Wx -= learning_rate * dWx
    Wh -= learning_rate * dWh
    b -= learning_rate * db
    Wy -= learning_rate * dWy
    by -= learning_rate * dby
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

y_preds, _ = forward_pass(X)
final_pred = y_preds[-1]
predicted_idx = np.argmax(final_pred)
predicted_word = idx_to_word[predicted_idx]

print(f"Input sequence: {input_sequence}")
print(f"Target word: {target_word}")
print(f"Predicted word: {predicted_word}")
print(f"Prediction probabilities: {[float('{:.4f}'.format(p)) for p in final_pred.flatten()]}")

