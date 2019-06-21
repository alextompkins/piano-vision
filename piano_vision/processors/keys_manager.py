import cv2
import numpy as np
from piano_vision.helpers import mean_and_standard_dev, apply_mask


class KeysManager:
	def __init__(self, ref_frame):
		self.ref_frame = ref_frame

		# Get key contours
		thresh = self.threshold(ref_frame)
		key_contours = self.find_key_contours(thresh)
		print('{} keys found'.format(len(key_contours)))
		# cv2.drawContours(ref_frame, key_contours, -1, (255, 0, 255), thickness=2)

		# Get a bounding rectangle for each key
		self.black_keys = tuple(map(cv2.boundingRect, key_contours))
		for rect in self.black_keys:
			x, y, w, h = rect
			cv2.rectangle(ref_frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

		mean_key_width = np.mean(tuple(map(lambda r: r[2], self.black_keys)))
		print(mean_key_width)

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
