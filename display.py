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
GREEN = (100, 200, 100)
RED = (200, 100, 100)
BUTTON_TEXT_COLOR = BLACK

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

# Function to generate a random digit and update information
def generate_random_digit():
    global idx, img, true_label, predicted_label, confidence
    idx = np.random.randint(0, test_images.shape[0])
    img = test_images[idx]
    true_label = test_labels[idx]
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_label = np.argmax(predictions)
    confidence = predictions[0][predicted_label]

# Function to generate a random incorrectly guessed digit and update information
def generate_incorrectly_guessed_digit():
    global idx, img, true_label, predicted_label, confidence
    incorrect_indices = np.where(test_labels != np.argmax(model.predict(test_images), axis=1))[0]
    if len(incorrect_indices) > 0:
        idx = np.random.choice(incorrect_indices)
        img = test_images[idx]
        true_label = test_labels[idx]
        img = np.expand_dims(img, axis=0)
        predictions = model.predict(img)
        predicted_label = np.argmax(predictions)
        confidence = predictions[0][predicted_label]

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the mouse click is within the buttons area
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if 160 <= mouse_x <= 240 and 320 <= mouse_y <= 360:  # New Digit button position adjusted
                generate_random_digit()
            elif 250 <= mouse_x <= 330 and 320 <= mouse_y <= 360:  # Incorrect button position adjusted
                generate_incorrectly_guessed_digit()

    # Display the image and predictions
    screen.fill(WHITE)
    img_surface = pygame.surfarray.make_surface(
        np.stack((np.rot90(np.flipud(img[0]), k=-1) * 255,) * 3, axis=-1))  # Flip and rotate, then convert to color
    img_surface = pygame.transform.scale(img_surface, (280, 280))  # Scale up the image
    screen.blit(img_surface, (60, 20))  # Center the image

    # Draw the buttons
    pygame.draw.rect(screen, GREEN, (160, 320, 80, 40))  # New Digit button position adjusted, Soft green color
    pygame.draw.rect(screen, RED, (250, 320, 80, 40))  # Incorrect button position adjusted, Soft red color
    font = pygame.font.Font(None, 24)
    button_text = font.render("New Digit", True, BUTTON_TEXT_COLOR)
    screen.blit(button_text, (163, 332))  # New Digit button text position adjusted
    button_text = font.render("Incorrect", True, BUTTON_TEXT_COLOR)
    screen.blit(button_text, (255, 332))  # Incorrect button text position adjusted

    font = pygame.font.Font(None, 24)
    true_label_text = font.render(f"True Label: {true_label}", True, BLACK)
    predicted_label_text = font.render(f"Predicted Label: {predicted_label}, Confidence: {confidence:.2f}", True, BLACK)
    screen.blit(true_label_text, (10, 350))
    screen.blit(predicted_label_text, (10, 380))

    pygame.display.flip()

pygame.quit()
