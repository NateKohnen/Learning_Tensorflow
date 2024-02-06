import pygame
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(_, _), (test_images, test_labels) = mnist.load_data()
test_images = test_images / 255.0

# Load the saved model
model = tf.keras.models.load_model("my_model.keras")

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Select a random image from the test set
idx = np.random.randint(0, test_images.shape[0])
img = test_images[idx]
true_label = test_labels[idx]

# Reshape the image to fit the model input shape
img = np.expand_dims(img, axis=0)

# Make a prediction with the loaded model
predictions = model.predict(img)
predicted_label = np.argmax(predictions)
confidence = predictions[0][predicted_label]

# Initialize the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("MNIST Prediction")

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Display the image and predictions
    screen.fill(WHITE)
    img_surface = pygame.surfarray.make_surface(
        np.stack((np.rot90(np.flipud(img[0]), k=-1) * 255,) * 3, axis=-1))  # Flip and rotate, then convert to color
    img_surface = pygame.transform.scale(img_surface, (280, 280))  # Scale up the image
    screen.blit(img_surface, (60, 20))  # Center the image

    font = pygame.font.Font(None, 24)
    true_label_text = font.render(f"True Label: {true_label}", True, BLACK)
    predicted_label_text = font.render(f"Predicted Label: {predicted_label}, Confidence: {confidence:.2f}", True, BLACK)
    screen.blit(true_label_text, (10, 320))
    screen.blit(predicted_label_text, (10, 350))

    pygame.display.flip()

pygame.quit()
