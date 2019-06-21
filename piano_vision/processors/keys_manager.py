import cv2
import numpy as np
from piano_vision.helpers import mean_and_standard_dev, apply_mask


class KeysManager:
	def __init__(self, ref_frame):
		self.ref_frame = ref_frame

		# Get black key contours
		thresh = self.threshold(ref_frame)
		key_contours = self.find_key_contours(thresh)
		# cv2.drawContours(ref_frame, key_contours, -1, (255, 0, 255), thickness=2)

		# Get a bounding rectangle for each black key
		self.black_keys = tuple(map(cv2.boundingRect, key_contours))
		print('{} black keys found'.format(len(self.black_keys)))
		for rect in self.black_keys:
			x, y, w, h = rect
			cv2.rectangle(ref_frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=1)

		# Get white keys, draw rectangles
		self.white_keys = tuple(self.find_white_keys(self.ref_frame))
		print('{} white keys found'.format(len(self.white_keys)))
		for rect in self.white_keys:
			x, y, w, h = rect
			cv2.rectangle(ref_frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=1)

		cv2.imshow('ref_frame', ref_frame)

	def threshold(self, frame):
		grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(grey, (3, 3), 0)
		thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 99, 40)
		return thresh

	def find_key_contours(self, thresholded):
		# Don't check for keys in the bottom quarter of the image
		h, w = thresholded.shape
		mask = np.zeros((h, w, 1), np.uint8)
		cv2.rectangle(mask, (0, 0), (w, h - round(h / 4)), color=255, thickness=cv2.FILLED)
		masked = apply_mask(thresholded, mask)

		# Discard contours that are more than 2 standard deviations below the mean
		contours, hierarchy = cv2.findContours(masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		mean, std_dev = mean_and_standard_dev(contours, key=cv2.contourArea)
		contours = tuple(filter(lambda c: cv2.contourArea(c) > (mean - 2 * std_dev), contours))

		return contours

	def find_white_keys(self, frame):
		height = frame.shape[0]
		cropped = frame[height - round(height / 3.5):height - round(height / 16)].copy()
		cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
		cropped = cv2.Canny(cropped, 10, 30)

		for row in cropped:
			for col, val in enumerate(row):
				if val and col < len(row) - 3 and row[col + 2]:
					row[col + 1] = val
					row[col + 2] = 0
					row[col] = 0
				if val and col < len(row) - 2 and row[col + 1]:
					row[col] = 0

		lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, threshold=2, minLineLength=5, maxLineGap=5)
		boundaries = {0}
		if lines is not None:
			for line in lines:
				for x1, y1, x2, y2 in line:
					boundaries.add(int(round((x1 + x2) / 2)))

		boundaries.add(cropped.shape[1])
		boundaries = sorted(boundaries)

		close = []
		keys = []
		for i in range(len(boundaries) - 1):
			start, end = boundaries[i:i+2]

			if abs(start - end) <= 5:
				close.append(start)
			else:
				if close:
					close.append(start)
					start = int(np.median(close))
					close = []
				keys.append((start, 0, end - start, frame.shape[0]))

		return keys
