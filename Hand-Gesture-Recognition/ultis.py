import pygame
import numpy as np
import cv2
def np2surface(image):
    s_image = pygame.surfarray.make_surface(np.swapaxes(image, 0, 1))
    return s_image

def letter_capture(char):
    return char

def addBorder(image):
    border_width = 10
    border_color = (0, 255, 0)  # Green color (BGR)