import cv2

from piano_vision.helpers import apply_mask, centre_of_contour
from piano_vision.processors import KeysManager


class PressedKeyDetector:
	MIN_CONTOUR_AREA = 150

	def __init__(self, ref_frame, keys_manager):
		self.ref_frame = ref_frame
		self.keys_manager: KeysManager = keys_manager

	def detect_pressed_keys(self, frame, skin_mask, fingertips=None):
		frame = frame.copy()
		# Dilate again to ensure that we don't include any small bits of skin
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
		dilated_mask = cv2.dilate(skin_mask, kernel, iterations=1)

		skin = apply_mask(frame, dilated_mask)
		frame = cv2.subtract(frame, skin)

		skin_ref = apply_mask(self.ref_frame, dilated_mask)
		ref = cv2.subtract(self.ref_frame, skin_ref)

		diff = self.get_diff(frame, ref)

		contours, hierarchy = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours = tuple(filter(lambda c: cv2.contourArea(c) > self.MIN_CONTOUR_AREA, contours))
		cv2.drawContours(frame, contours, -1, color=(0, 255, 0), thickness=cv2.FILLED)

		centres = tuple(map(centre_of_contour, contours))

		for centre in centres:
			cv2.circle(frame, (centre[0], centre[1]), radius=5, color=(0, 0, 255), thickness=cv2.FILLED)
		cv2.imshow('diff', diff)
		cv2.imshow('frame_with_diff', frame)

		pressed_keys = []
		for centre in centres:
			for key in [*self.keys_manager.white_keys, *self.keys_manager.black_keys]:
				if key.x < centre[0] < key.x + key.width and key.y < centre[1] < key.y + key.height:
					pressed_keys.append(key)

		if fingertips:
			pressed_keys = tuple(filter(lambda k:
				any(
					map(
						lambda f: k.x < f[0] < k.x + k.width and k.y < f[1] < k.y + k.height,
						fingertips
					)),
				pressed_keys))

		return pressed_keys

	@staticmethod
	def get_diff(frame, ref):
		diff = cv2.absdiff(frame, ref)
		diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
		# diff = cv2.GaussianBlur(diff, (3, 3), 0)
		diff = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 10)
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
		diff = cv2.dilate(diff, kernel, iterations=2)
		diff = cv2.erode(diff, kernel, iterations=2)
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
		diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel, iterations=1)
		diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel, iterations=1)
		return diff
