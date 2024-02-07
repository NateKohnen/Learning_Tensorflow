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
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BUTTON_COLOR = (150, 150, 150)
BUTTON_TEXT_COLOR = BLACK
NUMBER_COLOR = (70, 70, 70)

# Select a random image from the test set
idx = np.random.randint(0, test_images.shape[0])
img = test_images[idx]
true_label = test_labels[idx]

# Reshape the image to fit the model input shape
img = np.expand_dims(img, axis=0)

# Make a prediction with the loaded model
predictions = model.predict(img)
predicted_label = np.argmax(predictions)

# Initialize the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("MNIST Prediction")


# Function to generate a random digit and update displayed information
def generate_random_digit():
    global idx, img, true_label, predicted_label, predictions
    idx = np.random.randint(0, test_images.shape[0])
    img = test_images[idx]
    true_label = test_labels[idx]
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_label = np.argmax(predictions)


# Function to generate a random incorrectly guessed digit and update displayed information
def generate_incorrectly_guessed_digit():
    global idx, img, true_label, predicted_label, predictions
    incorrect_indices = np.where(test_labels != np.argmax(model.predict(test_images), axis=1))[0]
    if len(incorrect_indices) > 0:
        idx = np.random.choice(incorrect_indices)
        img = test_images[idx]
        true_label = test_labels[idx]
        img = np.expand_dims(img, axis=0)
        predictions = model.predict(img)
        predicted_label = np.argmax(predictions)


# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the mouse click is within the button's area
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if 160 <= mouse_x <= 240 and 320 <= mouse_y <= 360:  # 'New Digit' button click detection
                generate_random_digit()
            elif 250 <= mouse_x <= 330 and 320 <= mouse_y <= 360:  # 'Incorrect' button click detection
                generate_incorrectly_guessed_digit()

    # Display the image and predictions
    screen.fill(WHITE)
    img_surface = pygame.surfarray.make_surface(
        np.stack((np.rot90(np.flipud(img[0]), k=-1) * 255,) * 3, axis=-1))  # Flip and rotate the image
    img_surface = pygame.transform.scale(img_surface, (280, 280))  # Scale up the image
    screen.blit(img_surface, (60, 20))  # Center the image

    # Draw the buttons
    pygame.draw.rect(screen, BUTTON_COLOR, (160, 320, 80, 40))  # New Digit button position
    pygame.draw.rect(screen, BUTTON_COLOR, (250, 320, 80, 40))  # Incorrect button position
    font = pygame.font.Font(None, 24)
    button_text = font.render("New Digit", True, BUTTON_TEXT_COLOR)
    screen.blit(button_text, (163, 332))  # 'New Digit' button text
    button_text = font.render("Incorrect", True, BUTTON_TEXT_COLOR)
    screen.blit(button_text, (255, 332))  # 'Incorrect' button text

    # Draw the numbers and confidence values
    for i in range(10):
        number_text = font.render(str(i), True, NUMBER_COLOR)
        screen.blit(number_text, (400, 20 + i * 30))

        # Get confidence value for the current digit
        confidence = predictions[0][i]

        # Convert confidence value to a readable percentage
        confidence_percentage = f"{confidence * 100:.2f}%"

        # Render confidence value text
        confidence_text = font.render(confidence_percentage, True, BLACK)
        screen.blit(confidence_text, (500, 20 + i * 30))

    # Display true and guessed labels
    true_label_text = font.render(f"True Value: {true_label}", True, BLACK)
    predicted_label_text = font.render(f"Guessed Value: {predicted_label}", True, BLACK)
    screen.blit(true_label_text, (400, 320))
    screen.blit(predicted_label_text, (400, 350))

    pygame.display.flip()

pygame.quit()
