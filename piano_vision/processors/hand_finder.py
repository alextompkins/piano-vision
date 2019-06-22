import cv2
import numpy as np

from piano_vision.helpers import avg_of_groups, index_of_closest, group, dist


class HandFinder:
	# THRESHOLDS
	# define the upper and lower boundaries of the HSV pixel
	# intensities to be considered 'skin'
	SKIN_LOWER = np.array([0, 48, 80], dtype='uint8')
	SKIN_UPPER = np.array([200, 255, 255], dtype='uint8')
	# define the parameters that constitute a hand
	MIN_CONTOUR_AREA = 150
	MAX_DIST = 30
	ANGLE_MAX = 180

	def get_skin_mask(self, frame):
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		skin_mask = cv2.inRange(hsv, self.SKIN_LOWER, self.SKIN_UPPER)

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
		skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
		skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)
		return skin_mask

	def get_hand_contours(self, skin_mask):
		# skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
		contours, hierarchy = cv2.findContours(skin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		largest_contours = sorted(contours, key=cv2.contourArea)[:-7:-1]  # 3 contours per hand

		largest_contours = filter(lambda c: cv2.contourArea(c) > self.MIN_CONTOUR_AREA, largest_contours)
		return tuple(largest_contours)

	def find_fingertips(self, hand_contours):
		hands = []
		convexity_defects = []

		for contour in hand_contours:
			convex_pts = cv2.convexHull(contour)
			group_averages = np.array(avg_of_groups(group(convex_pts, self.MAX_DIST)))
			closest_convex_pts = np.array(index_of_closest(contour, group_averages))
			defects = cv2.convexityDefects(contour, closest_convex_pts)
			if defects is None:
				defects = []
			convexity_defects.append(defects)

		for i, hand_defects in enumerate(convexity_defects):
			contour = hand_contours[i]
			fingertips = []
			for j, defects in enumerate(hand_defects):
				s = defects[0][0]
				e = defects[0][1]
				f = defects[0][2]
				d = defects[0][3]

				start = tuple(contour[s][0])
				end = tuple(contour[e][0])
				far = tuple(contour[f][0])

				a = dist([start], [end])
				b = dist([far], [start])
				c = dist([far], [end])

				angle_deg = np.degrees(np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)))

				if angle_deg < self.ANGLE_MAX:
					if start not in fingertips:
						fingertips.append(start)
					if end not in fingertips:
						fingertips.append(end)

			hands.append(fingertips)

		return hands

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
