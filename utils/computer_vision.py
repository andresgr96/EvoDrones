import cv2
import numpy as np

# Displays the image feedback given an RGB image
def display_drone_image(rgb_image):
    cv2.imshow('Drone Camera Image', rgb_image)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()

# Computes binary mask for red color
def red_mask(img):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 50, 70), (9, 255, 255))

    return mask


# Segments image into 9 segments for visual guidance
def segment_image(img):
    color = (0, 0, 255)  # red
    thickness = 2
    num_segments = 3
    segment_width = img.shape[1] // num_segments
    segment_height = img.shape[0] // num_segments

    for i in range(1, num_segments):
        x_position = i * segment_width
        cv2.line(img, (x_position, 0), (x_position, img.shape[0]), color, thickness)

    for i in range(1, num_segments):
        y_position = i * segment_height
        cv2.line(img, (0, y_position), (img.shape[1], y_position), color, thickness)

    return img
