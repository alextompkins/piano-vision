import cv2
import numpy as np
from math import atan, degrees


class PianoEdgeFinder:
	NUM_CONTOURS = 1

	def process_frame(self, frame):
		self.find_white_area(frame)
		grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imshow('grey', grey)
		edges = cv2.Canny(grey, 100, 200)
		cv2.imshow('canny', edges)
		lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=100, maxLineGap=50)
		angles = []
		if lines is not None:
			for line in lines:
				for x1, y1, x2, y2 in line:
					angle = degrees(atan((y2 - y1) / (x2 - x1)))
					angles.append(angle)
					cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
		print(angles[int(len(angles) / 2)])
		cv2.imshow('edges', frame)

	def find_white_area(self, frame):
		frame = frame.copy()
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		white = cv2.inRange(hsv, np.array([0, 0, 240]), np.array([255, 30, 255]))

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
		white = cv2.dilate(white, kernel, iterations=3)
		white = cv2.erode(white, kernel, iterations=3)

		contours, hierarchy = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		largest_contours = sorted(contours, key=cv2.contourArea)[:(-1 - self.NUM_CONTOURS):-1]  # Get n largest contours
		cv2.drawContours(frame, largest_contours, -1, (255, 0, 255), thickness=cv2.FILLED)
		rects = []
		for contour in largest_contours:
			x, y, w, h = cv2.boundingRect(contour)
			rect = ((x, y), (x + w, y + h))
			rects.append(rect)
			cv2.rectangle(white, rect[0], rect[1], 255, thickness=2)

		min_y = average_rounded(r[0][1] for r in rects)
		max_y = average_rounded(r[1][1] for r in rects)
		min_x = min(r[0][0] for r in rects)
		max_x = max(r[1][0] for r in rects)
		cv2.rectangle(white, (min_x, min_y), (max_x, max_y), 255, 2)

		corners_pre = np.float32([[min_x, min_y], [max_x, min_y], [min_x, max_y], [max_x, max_y]])
		corners_post = np.float32([[0, 0], [max_x - min_x, 0], [0, max_y - min_y], [max_x - min_x, max_y - min_y]])
		matrix = cv2.getPerspectiveTransform(corners_pre, corners_post)
		warped = cv2.warpPerspective(white, matrix, (max_x - min_x, max_y - min_y))

		cv2.imshow('white', white)
		cv2.imshow('warped', warped)


def average_rounded(num_iter):
	total = 0
	length = 0
	for i in num_iter:
		total += i
		length += 1

	return round(total / length)
