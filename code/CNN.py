import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from typing import Tuple, List
import pickle

class ConvLayer:
    def __init__(self, num_filters: int, filter_size: int, stride: int, padding: int, input_channels: int = 3):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        # Initialize filters with random values - match input channel count
        self.filters = np.random.randn(num_filters, filter_size, filter_size, input_channels) * 0.1
        self.biases = np.zeros(num_filters)
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        # Get dimensions
        batch_size, h_in, w_in, channels = input_data.shape
        
        # Calculate output dimensions
        h_out = (h_in + 2 * self.padding - self.filter_size) // self.stride + 1
        w_out = (w_in + 2 * self.padding - self.filter_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, h_out, w_out, self.num_filters))
        
        # Add padding if needed
        if self.padding > 0:
            padded_input = np.pad(input_data, 
                                 ((0, 0), (self.padding, self.padding), 
                                  (self.padding, self.padding), (0, 0)), 
                                 'constant')
        else:
            padded_input = input_data
            
        # Perform convolution
        for i in range(batch_size):
            for h in range(h_out):
                for w in range(w_out):
                    for f in range(self.num_filters):
                        h_start = h * self.stride
                        h_end = h_start + self.filter_size
                        w_start = w * self.stride
                        w_end = w_start + self.filter_size
                        
                        input_slice = padded_input[i, h_start:h_end, w_start:w_end, :]
                        output[i, h, w, f] = np.sum(input_slice * self.filters[f]) + self.biases[f]
        
        # Apply ReLU activation
        return np.maximum(0, output)

class MaxPoolLayer:
    def __init__(self, pool_size: int, stride: int):
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        batch_size, h_in, w_in, channels = input_data.shape
        
        h_out = (h_in - self.pool_size) // self.stride + 1
        w_out = (w_in - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, h_out, w_out, channels))
        
        for i in range(batch_size):
            for h in range(h_out):
                for w in range(w_out):
                    for c in range(channels):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        
                        input_slice = input_data[i, h_start:h_end, w_start:w_end, c]
                        output[i, h, w, c] = np.max(input_slice)
        
        return output

class FlattenLayer:
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        batch_size = input_data.shape[0]
        flattened_size = np.prod(input_data.shape[1:])
        return input_data.reshape(batch_size, flattened_size)

class DenseLayer:
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros(output_size)
        self.activation = activation
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        linear_output = np.dot(input_data, self.weights) + self.biases
        
        if self.activation == 'relu':
            return np.maximum(0, linear_output)
        elif self.activation == 'softmax':
            exp_values = np.exp(linear_output - np.max(linear_output, axis=1, keepdims=True))
            return exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return linear_output

class CNN:
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int = 10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.build_model()
    
    def build_model(self):
        # Create layers according to the specified architecture
        self.layers = []
        # Conv1: 8 filters (4×4), stride=2, padding=1, input channels=3 (RGB)
        self.layers.append(ConvLayer(num_filters=8, filter_size=4, stride=2, padding=1, input_channels=3))
        # MaxPool1: 2×2, stride=2
        self.layers.append(MaxPoolLayer(pool_size=2, stride=2))
        # Conv2: 16 filters (3×3), stride=1, no padding, input channels=8 (from previous conv)
        self.layers.append(ConvLayer(num_filters=16, filter_size=3, stride=1, padding=0, input_channels=8))
        # MaxPool2: 2×2, stride=2
        self.layers.append(MaxPoolLayer(pool_size=2, stride=2))
        # Flatten
        self.layers.append(FlattenLayer())
        
        # Calculate input size for the dense layer
        # After first conv (32x32x3 -> 16x16x8) and pool (16x16x8 -> 8x8x8)
        # After second conv (8x8x8 -> 6x6x16) and pool (6x6x16 -> 3x3x16)
        # Flattened: 3*3*16 = 144
        self.layers.append(DenseLayer(input_size=144, output_size=128))
        self.layers.append(DenseLayer(input_size=128, output_size=self.num_classes, activation='softmax'))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        # Forward pass through all layers
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
    
    def summary(self):
        print("CNN Architecture:")
        print(f"Input shape: {self.input_shape}")
        print("Layers:")
        print("1. Conv2D: 8 filters (4×4), stride=2, padding=1, ReLU")
        print("2. MaxPooling2D: (2×2), stride=2")
        print("3. Conv2D: 16 filters (3×3), stride=1, no padding, ReLU")
        print("4. MaxPooling2D: (2×2), stride=2")
        print("5. Flatten")
        print("6. Dense: 128 neurons, ReLU")
        print(f"7. Dense: {self.num_classes} neurons, Softmax")

def preprocess_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path)
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

if __name__ == "__main__":
    # Create the model
    model = CNN(input_shape=(32, 32, 3), num_classes=10)
    model.summary()
    
    # Load and preprocess image
    image_path = r"C:\outDesktop\Collage\Computational Neuroscience\code\images\ROG_Branding_Wallpaper_RETRO RE imagined_1920x1080_single.jpg"
    
    if os.path.exists(image_path):
        img_data = preprocess_image(image_path)
        
        # Display the image
        plt.imshow(img_data[0])
        plt.title('Input Image (32x32)')
        plt.axis('off')
        plt.show()
        
        # Make prediction
        prediction = model.predict(img_data)
        predicted_class = np.argmax(prediction, axis=1)[0]
        print(f"Sample prediction (untrained model): Class {predicted_class}")
        print(f"Prediction probabilities: {prediction[0]}")
        print("\nNote: Model is untrained, predictions are random.")
    else:
        print(f"Error: Could not find image at {image_path}")
