import pygame
import sys
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model("my_model.keras")

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 280, 280
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Illustration Pad")

# Colors
black = (0, 0, 0)
white = (255, 255, 255)

# Drawing variables
drawing = False
last_pos = (0, 0)
radius = 8

# Analyze the drawn image
def analyze_image():
    # Load the drawn image
    drawn_image = Image.open("drawn_image.png")

    # Convert to grayscale
    drawn_image = drawn_image.convert("L")

    # Resize the image to 28x28 pixels
    drawn_image = drawn_image.resize((28, 28))

    # Save the version of the image that will be fed to the model
    drawn_image.save("modified_image.png")

    # Convert to a NumPy array
    image_data = np.array(drawn_image)

    # Reshape and preprocess the image for model input
    img = image_data.reshape(1, 28, 28) / 255.0

    # Make predictions using the model
    predictions = model.predict(img)
    predicted_label = np.argmax(predictions)

    # Print the predicted label and confidence values
    print("Predicted Label:", predicted_label)
    print("Confidence Values:", predictions)

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            last_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION and drawing:
            pygame.draw.line(screen, white, last_pos, event.pos, radius * 2)
            last_pos = event.pos
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                # Save the drawn image
                pygame.image.save(screen, "drawn_image.png")
                print("Image saved.")
            elif event.key == pygame.K_a:
                # Analyze the drawn image using the model
                analyze_image()
            elif event.key == pygame.K_d:
                # Analyze the drawn image using the model
                screen.fill(black)

    # Update the display
    pygame.display.flip()

    # Limit frames per second
    pygame.time.Clock().tick(60)
