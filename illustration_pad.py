import pygame
import sys
import numpy as np
from PIL import Image

# Initialize Pygame
pygame.init()

# Set up the screen
width, height = 280, 280
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Simple Drawing App")

# Set up colors
black = (0, 0, 0)
white = (255, 255, 255)

# Set up drawing variables
drawing = False
last_pos = (0, 0)
radius = 5

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

    # Update the display
    pygame.display.flip()

    # Limit frames per second
    pygame.time.Clock().tick(60)

    # Save the drawn image when 's' key is pressed
    keys = pygame.key.get_pressed()
    if keys[pygame.K_s]:
        # Capture the screen and convert it to a NumPy array
        image_data = pygame.surfarray.array3d(screen)
        image_data = np.flip(image_data, axis=0)  # Flip vertically

        # Convert to grayscale
        image_data = np.dot(image_data[..., :3], [0.299, 0.587, 0.114])

        # Convert to a PIL Image
        pil_image = Image.fromarray(image_data.astype(np.uint8))

        # Resize the image to 28x28 pixels
        pil_image = pil_image.resize((28, 28))

        # Save the image as a PNG file
        pil_image.save("drawn_image.png")

        # Rotate the saved image by 270 degrees
        rotated_image = pil_image.rotate(270)

        # Save the rotated image
        rotated_image.save("drawn_image_rotated.png")
