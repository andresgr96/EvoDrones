import cv2
import numpy as np


# Displays the image feedback given an RGB image
def display_drone_image(rgb_image):
    # if not isinstance(rgb_image, np.ndarray):
    #     rgb_image = np.array(rgb_image)

    if rgb_image.dtype != np.uint8:
        rgb_image = (rgb_image * 255).clip(0, 255).astype(np.uint8)  # Convert to uint8 and clip values

    cv2.imshow('Drone Camera Image', cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
    # cv2.destroyAllWindows()


# Computes binary mask for red color
def red_mask(img):
    # Ensure img is in uint8 format
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
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


def detect_objects(masked_image):
    num_segments = 3

    # Define the segments
    segments = []
    segment_width = masked_image.shape[1] // num_segments
    segment_height = masked_image.shape[0] // num_segments
    # print(segment_width, segment_height)
    # segment_names = ["Top Left", "Middle Left", "Bottom Left", "Top Center", "Middle Center", "Bottom Center",
    #                  "Top Right", "Middle Right", "Bottom Right"]

    # A cleaner way to define each segment
    for i in range(num_segments):
        for j in range(num_segments):
            start_x = i * segment_width
            end_x = (i + 1) * segment_width
            start_y = j * segment_height
            end_y = (j + 1) * segment_height
            segments.append((start_x, end_x, start_y, end_y))

    # Count pixels in each segment and form output
    segment_pixel_counts = [0] * len(segments)
    for idx, (start_x, end_x, start_y, end_y) in enumerate(segments):
        segment = masked_image[start_y:end_y, start_x:end_x]
        segment_pixel_count = cv2.countNonZero(segment)

        # Normalize the value so the NN likes it
        max_pixel_count = (segment_width * segment_height)
        normalized_pixel_count = segment_pixel_count / max_pixel_count
        segment_pixel_counts[idx] = normalized_pixel_count
        # segment_name = segment_names[idx]
        # print(f"Normalized Pixel Count for {segment_name} segment: {normalized_pixel_count}")  # For testing

    return segment_pixel_counts
