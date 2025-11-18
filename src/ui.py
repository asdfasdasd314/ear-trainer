import pygame
pygame.init()

screen_width = 800
screen_height = 600
screen_size = (screen_width, screen_height)

screen = pygame.display.set_mode(screen_size)

pygame.display.set_caption("Chord Classifier")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Drawing commands go here ---
    screen.fill((0, 0, 0)) # Fill the screen with black

    pygame.display.flip() # Update the full display Surface to the screen

pygame.quit() # Uninitialize Pygame modules and exit
