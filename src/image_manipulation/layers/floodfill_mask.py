import cv2

import numpy as np


# noinspection PyTypeChecker
def generate_mask(img: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY_INV)
    binary = binary.astype(np.uint8)
    h, w = binary.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    starting_points = [(0, 0), (0, h-1), (w-1, 0), (w-1, h-1),
                       (w//2, 0), (w//2, h-1), (0, h//2), (w-1, h//2)]

    for sp in starting_points:
        cv2.floodFill(binary, mask, sp, 255)

    mask = cv2.bitwise_not(binary)

    return mask
