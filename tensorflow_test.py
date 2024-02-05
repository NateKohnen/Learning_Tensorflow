import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 pixel grid
    layers.Dense(128, activation='relu'),  # Hidden layer with ReLU activation
    layers.Dense(10, activation='softmax')  # Output layer with softmax activation
])

# Compile the model
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Check if there is a saved model file, and load it if exists
try:
    model.load_weights("mnist_model.h5")
    print("Model loaded successfully.")
except:
    # Train the model if no saved weights are found
    print("Training the model...")
    model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

    # Save the trained model weights
    model.save_weights("mnist_model.h5")
    print("Model saved.")

# Function to display a single image along with model prediction
def visualize_single_prediction(index, correct_only, digit_filter, images, labels, model):
    image = images[index]
    label = labels[index]

    # Reshape the image to (1, 28, 28) for model prediction
    input_image = np.expand_dims(image, axis=0)

    # Make a prediction
    predictions = model.predict(input_image)
    predicted_label = np.argmax(predictions)

    # Check if the prediction is correct based on the filter
    prediction_correct = (predicted_label == label)

    # Check if the image should be displayed based on the filters
    display_image = (not correct_only or prediction_correct) and (digit_filter is None or label == digit_filter)

    if display_image:
        # Display the image, true label, predicted label, and confidence
        plt.imshow(image, cmap='gray')
        plt.title(f'True: {label}\nPredicted: {predicted_label}\nConfidence: {predictions[0][predicted_label]:.2f}')
        plt.axis('off')
        plt.show()

# Create interactive controls for digit navigation and filtering
interact(visualize_single_prediction,
         index=widgets.IntSlider(min=0, max=len(test_images)-1, step=1, value=0),
         correct_only=widgets.Checkbox(value=False, description='Correct Predictions Only'),
         digit_filter=widgets.Dropdown(options=[None] + list(range(10)), value=None, description='Digit Filter'),
         images=widgets.fixed(test_images),
         labels=widgets.fixed(test_labels),
         model=widgets.fixed(model))
