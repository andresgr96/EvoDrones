import cv2
import numpy as np

# Displays the image feedback given an RGB image
def display_drone_image(rgb_image):
    cv2.imshow('Drone Camera Image', rgb_image)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()