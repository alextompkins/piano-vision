import cv2
import numpy as np


class HandFinder:
	# define the upper and lower boundaries of the HSV pixel
	# intensities to be considered 'skin'
	lower = np.array([0, 48, 80], dtype='uint8')
	upper = np.array([200, 255, 255], dtype='uint8')

	def get_skin_mask(self, frame):
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		skin_mask = cv2.inRange(hsv, self.lower, self.upper)

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
		skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
		skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)
		return skin_mask

	def get_hand_contours(self, skin_mask):
		# skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
		contours, hierarchy = cv2.findContours(skin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		largest_contours = sorted(contours, key=cv2.contourArea)[:-3:-1]
		return largest_contours

	def process_frame(self, frame):
		# cv2.drawContours(frame, largest_contours, -1, (255, 0, 255), thickness=cv2.FILLED)
		# cv2.connectedComponents(skin_mask, skin)
		pass

	def find_skeleton(self, frame):
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		size = np.size(frame)
		skel = np.zeros(frame.shape, np.uint8)

		ret, frame = cv2.threshold(frame, 100, 255, 0)
		element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
		done = False

		while not done:
			eroded = cv2.erode(frame, element)
			temp = cv2.dilate(eroded, element)
			temp = cv2.subtract(frame, temp)
			skel = cv2.bitwise_or(skel, temp)
			frame = eroded.copy()

			zeros = size - cv2.countNonZero(frame)
			if zeros == size:
				done = True

		return skel
