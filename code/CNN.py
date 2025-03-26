import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Function to build the CNN model
def build_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential()
    
    # First Convolution Layer: 8 filters of size 4×4, stride=2, padding=1, ReLU
    model.add(layers.Conv2D(8, (4, 4), strides=2, padding='same', 
                            activation='relu', input_shape=input_shape))
    
    # First Max Pooling Layer: 2×2 pool size, stride=2
    model.add(layers.MaxPooling2D((2, 2), strides=2))
    
    # Second Convolution Layer: 16 filters of size 3×3, stride=1, no padding, ReLU
    model.add(layers.Conv2D(16, (3, 3), strides=1, padding='valid', 
                            activation='relu'))
    
    # Second Max Pooling Layer: 2×2 pool size, stride=2
    model.add(layers.MaxPooling2D((2, 2), strides=2))
    
    # Flatten Layer
    model.add(layers.Flatten())
    
    # Fully connected layer with 128 neurons and ReLU activation
    model.add(layers.Dense(128, activation='relu'))
    
    # Output layer with softmax activation for 10 classes
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# Function to preprocess an image
def preprocess_image(image_path):
    # Load image
    img = Image.open(image_path)
    
    # Resize to 32x32
    img = img.resize((32, 32))
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Main execution
if __name__ == "__main__":
    # Build the model
    model = build_cnn_model()
    
    # Print model summary
    model.summary()
    
    # Compile the model with categorical cross-entropy loss and Adam optimizer
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Load and preprocess the sample image
    image_path = r"C:\outDesktop\Collage\Computational Neuroscience\code\images\ROG_Branding_Wallpaper_RETRO RE imagined_1920x1080_single.jpg"
    
    if os.path.exists(image_path):
        # Preprocess the image
        img_data = preprocess_image(image_path)
        
        # Display the image
        plt.imshow(img_data[0])
        plt.title('Input Image (Resized to 32x32)')
        plt.axis('off')
        plt.show()
        
        # Make a sample prediction
        # Note: Without training, this is just a random prediction
        prediction = model.predict(img_data)
        predicted_class = np.argmax(prediction, axis=1)[0]
        print(f"Sample prediction (untrained model): Class {predicted_class}")
        print(f"Prediction probabilities: {prediction[0]}")
        
        print("\nNote: Since the model is untrained, predictions are random.")
        print("To use this model properly, you would need to train it on labeled data first.")
    else:
        print(f"Error: Could not find image at {image_path}")
        print("Please ensure the image exists at the specified path.")
