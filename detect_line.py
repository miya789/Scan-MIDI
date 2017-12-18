import cv2
import logging
import os
import math

def load_image(filename):
    logging.info('Image file: %s', filename)
    image = cv2.imread(filename)
    if image is None:
        raise Exception('Failed to read image file: %s' % filename)
    return image

def detect_lines(image, original_image):
    lines = cv2.HoughLinesP(image, rho=5, theta=math.pi / 180.0 * 90,
                            threshold=200, minLineLength=30, maxLineGap=5)
    # Draw detected segments on the original image.
    if lines is not None:
        for (x1, y1, x2, y2) in lines[0]:
            cv2.line(original_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

image = load_image('9418699869_396bc62c3b_o.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

retval, binarized = cv2.threshold(gray, 224, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('binarized', binarized)

detect_lines(binarized, image)
cv2.imshow('score', image)
cv2.waitKey()
