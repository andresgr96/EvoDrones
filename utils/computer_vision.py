import cv2
import numpy as np

# Displays the image feedback given an RGB image
def display_drone_image(rgb_image):
    cv2.imshow('Drone Camera Image', rgb_image)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()

def red_mask(img):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 50, 70), (9, 255, 255))

    return mask
