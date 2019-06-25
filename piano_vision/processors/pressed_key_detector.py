import cv2

from piano_vision.helpers import apply_mask, centre_of_contour
from piano_vision.processors import KeysManager


class PressedKeyDetector:
	MIN_CONTOUR_AREA = 100
	STICKINESS = 2

	def __init__(self, ref_frame, keys_manager):
		self.ref_frame = ref_frame
		self.keys_manager: KeysManager = keys_manager
		self.currently_pressed = set()
		self.to_be_added = dict()
		self.to_be_removed = dict()

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
		# cv2.imshow('diff', diff)
		# cv2.imshow('frame_with_diff', frame)

		pressed_keys = set()
		for centre in centres:
			for key in [*self.keys_manager.white_keys, *self.keys_manager.black_keys]:
				if key.x < centre[0] < key.x + key.width and key.y < centre[1] < key.y + key.height:
					pressed_keys.add(key)

		if fingertips:
			pressed_keys = tuple(filter(
				# Filter pressed keys to only those which contain a fingertip
				lambda k: any(map(lambda f: self.fingertip_within_key(f, k), fingertips)),
				pressed_keys
			))

		self.process_sticky_pressed_changes(pressed_keys)
		return self.currently_pressed

	def process_sticky_pressed_changes(self, pressed_keys):
		# If a key was going to be added but is no longer pressed, remove from to_be_added
		delete_after = []
		for key in self.to_be_added:
			if key not in pressed_keys:
				delete_after.append(key)
		for key in delete_after:
			del self.to_be_added[key]

		for key in pressed_keys:
			# If a key was going to be removed but is now pressed again, delete from to_be_removed
			if key in self.to_be_removed:
				del self.to_be_removed[key]
			if key not in self.currently_pressed:
				# If the key is set to be added and is still pressed, reduce its counter by 1
				if key in self.to_be_added:
					self.to_be_added[key] -= 1
					# Key is ready to be added
					if self.to_be_added[key] == 0:
						del self.to_be_added[key]
						self.currently_pressed.add(key)
				# Otherwise if we haven't seen this key before, add it in future
				else:
					self.to_be_added[key] = self.STICKINESS

		delete_after = []
		for key in self.currently_pressed:
			if key not in pressed_keys:
				if key in self.to_be_removed:
					self.to_be_removed[key] -= 1
					if self.to_be_removed[key] == 0:
						del self.to_be_removed[key]
						delete_after.append(key)
				else:
					self.to_be_removed[key] = self.STICKINESS
		for key in delete_after:
			self.currently_pressed.remove(key)

	@staticmethod
	def fingertip_within_key(fingertip, key):
		return key.x < fingertip[0] < (key.x + key.width) and key.y < fingertip[1] < (key.y + key.height)

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
