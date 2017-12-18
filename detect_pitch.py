import cv2
import logging
import os
import math
import numpy

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

def pattern_match(image, original_image):
    template_width = 15
    template_height = 15
    ellipse = numpy.zeros((template_height, template_width, 1),
                            numpy.uint8)
    cv2.ellipse(ellipse, center=(8, 8), axes=(3, 5), angle=60,
                startAngle=0, endAngle=360, color=255, thickness=-1,
                lineType=cv2.CV_AA)
    cv2.imshow('ellipse', ellipse)

    matches = cv2.matchTemplate(image, ellipse, cv2.cv.CV_TM_CCORR_NORMED)

    threshold = 0.73
    for y in xrange(matches.shape[0]):
        for x in xrange(matches.shape[1]):
            if matches[y][x] > threshold:
                cv2.rectangle(original_image, (x, y),
                                (x + template_width, y + template_height),
                                (255, 0, 0), 1)

def detect_pitches(image, original_image):
    classifier = cv2.CascadeClassifier(os.path.join(
            'TrainingAssistant', 'pitch', 'cascade.xml'))
    clefs = classifier.detectMultiScale(image, 1.01, 2)

    for x, y, width, height in clefs:
        cv2.rectangle(original_image, (x, y), (x + width, y + height),
                    (0, 0, 255), 1)

image = load_image('bunbunbuncolor.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gaus = cv2.GaussianBlur(image, (3,3), 1)

retval, binarized = cv2.threshold(gray, 224, 255, cv2.THRESH_BINARY_INV)

detect_pitches(gaus, image)
cv2.imshow('score', image)
cv2.waitKey()
