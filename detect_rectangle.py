import cv2
import numpy
import logging

def load_image(filename):
    logging.info('Image file: %s', filename)
    image = cv2.imread(filename)
    if image is None:
        raise Exception('Failed to read image file: %s' % filename)
    return image

def transform_by4(img, points):

	points = sorted(points, key=lambda x:x[1])
	bottom = sorted(points[2:], key=lambda x:x[0], reverse=True)
	top = sorted(points[:2], key=lambda x:x[0])
	points = numpy.array(top + bottom, dtype='float32')

	width = max(numpy.sqrt(((points[0][0]-points[2][0])**2)*2), numpy.sqrt(((points[1][0]-points[3][0])**2)*2))
	height = max(numpy.sqrt(((points[0][1]-points[2][1])**2)*2), numpy.sqrt(((points[1][1]-points[3][1])**2)*2))

	dst = numpy.array([
			numpy.array([0, 0]),
			numpy.array([width-1, 0]),
			numpy.array([width-1, height-1]),
			numpy.array([0, height-1]),
			], numpy.float32)

	trans = cv2.getPerspectiveTransform(points, dst)
	return cv2.warpPerspective(img, trans, (int(width), int(height)))


if __name__ == '__main__':
	cam = cv2.VideoCapture(0)

	while cv2.waitKey(10) == -1:
		orig = load_image('example.png')

		lines = orig.copy()


		canny = cv2.cvtColor(orig, cv2.cv.CV_BGR2GRAY)
		canny = cv2.GaussianBlur(canny, (5, 5), 0)
		canny = cv2.Canny(canny, 50, 100)
		cv2.imshow('canny', canny)

		cnts = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
		cnts.sort(key=cv2.contourArea, reverse=True)

		warp = None
		for i, c in enumerate(cnts):
			arclen = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02*arclen, True)

			level = 1 - float(i)/len(cnts)
			if len(approx) == 4:
				cv2.drawContours(lines, [approx], -1, (0, 0, 255*level), 2)
				if warp == None:
					warp = approx.copy()
			else:
				cv2.drawContours(lines, [approx], -1, (0, 255*level, 0), 2)

			for pos in approx:
				cv2.circle(lines, tuple(pos[0]), 4, (255*level, 0, 0))

		cv2.imshow('edge', lines)

		if warp != None:
			warped = transform_by4(orig, warp[:,0,:])
			cv2.imshow('warp', warped)

	cam.release()
	cv2.destroyAllWindows()
