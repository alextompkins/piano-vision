import cv2
import numpy as np
from math import atan, degrees


class KeyboardBounder:
	def find_rotation(self, frame) -> float:
		frame = frame.copy()
		grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(grey, 100, 200)
		lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=100, maxLineGap=50)

		angles = []
		if lines is not None:
			for line in lines:
				for x1, y1, x2, y2 in line:
					angle = degrees(atan((y2 - y1) / (x2 - x1)))
					angles.append(angle)
					# cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
		return angles[int(len(angles) / 2)]  # return median angle

	def find_bounds(self, frame):
		frame = frame.copy()
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		white = cv2.inRange(hsv, np.array([0, 0, 240]), np.array([255, 30, 255]))

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
		white = cv2.dilate(white, kernel, iterations=3)
		white = cv2.erode(white, kernel, iterations=5)
		white = cv2.dilate(white, kernel, iterations=2)

		cv2.imshow('white', white)

		contours, hierarchy = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		largest_contour = max(contours, key=cv2.contourArea)
		# cv2.drawContours(frame, largest_contour, -1, (255, 0, 255), thickness=cv2.FILLED)

		x, y, w, h = cv2.boundingRect(largest_contour)
		return [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]

	def get_bounded_section(self, frame, bounds):
		min_x, max_x, min_y, max_y = bounds[0][0], bounds[1][0], bounds[0][1], bounds[2][1]

		corners_pre = np.float32([[min_x, min_y], [max_x, min_y], [min_x, max_y], [max_x, max_y]])
		corners_post = np.float32([[0, 0], [max_x - min_x, 0], [0, max_y - min_y], [max_x - min_x, max_y - min_y]])
		matrix = cv2.getPerspectiveTransform(corners_pre, corners_post)
		return cv2.warpPerspective(frame, matrix, (max_x - min_x, max_y - min_y))
