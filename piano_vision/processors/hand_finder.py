import cv2
import numpy as np


class HandFinder:
	# define the upper and lower boundaries of the HSV pixel
	# intensities to be considered 'skin'
	lower = np.array([0, 48, 80], dtype='uint8')
	upper = np.array([200, 255, 255], dtype='uint8')

	def process_frame(self, frame):
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		skin_mask = cv2.inRange(hsv, self.lower, self.upper)

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
		skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
		skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
		cv2.imshow('skin_mask', skin_mask)

		contours, hierarchy = cv2.findContours(skin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		largest_contours = sorted(contours, key=cv2.contourArea)[:-3:-1]
		cv2.drawContours(frame, largest_contours, -1, (255, 0, 255), thickness=2)

		skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
		skin = cv2.bitwise_and(frame, frame, mask=skin_mask)

		cv2.imshow('skin', np.vstack([frame, skin]))
