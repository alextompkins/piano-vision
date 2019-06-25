import cv2
import numpy as np
from enum import Enum
from piano_vision.helpers import mean_and_standard_dev, apply_mask


class Note(Enum):
	A = 0
	A_SHARP = 0.5
	B = 1
	C = 2
	C_SHARP = 2.5
	D = 3
	D_SHARP = 3.5
	E = 4
	F = 5
	F_SHARP = 5.5
	G = 6
	G_SHARP = 6.5

	def pretty_name(self):
		return self.name.replace('_SHARP', '#')


class Key:
	def __init__(self, x, y, width, height, note=None, octave=None):
		self.x = x
		self.y = y
		self.width = width
		self.height = height
		self.note = note
		self.octave = octave

	def __repr__(self) -> str:
		return 'Key(note={}, octave={}, x={})'.format(self.note, self.octave, self.x)

	def __str__(self) -> str:
		if self.note:
			return '{}{}'.format(self.note.pretty_name(), self.octave)
		else:
			return '??'


class KeysManager:
	def __init__(self, ref_frame):
		self.ref_frame = ref_frame

		# Get black key contours
		thresh = self.threshold(ref_frame)
		# cv2.imshow('black_keys_thresholded', thresh)
		key_contours = self.find_key_contours(thresh)

		display_frame = self.ref_frame.copy()
		cv2.drawContours(display_frame, key_contours, -1, (255, 0, 255), thickness=1)
		# cv2.imshow('black_keys_contours', display_frame)

		# Get a bounding rectangle for each black key
		self.black_keys = list(map(lambda c: Key(*cv2.boundingRect(c)), key_contours))
		self.white_keys = list(map(lambda r: Key(*r), self.find_white_keys(self.ref_frame)))

		self.label_keys()

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
		# cv2.imshow('white_key_edges_pre', cropped)

		for row in cropped:
			for col, val in enumerate(row):
				if val and col < len(row) - 3 and row[col + 2]:
					row[col + 1] = val
					row[col + 2] = 0
					row[col] = 0
				if val and col < len(row) - 2 and row[col + 1]:
					row[col] = 0

		# cv2.imshow('white_key_edges_post', cropped)

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

	def label_keys(self):
		self.black_keys.sort(key=lambda k: k.x)
		distances = []
		for i, key in tuple(enumerate(self.black_keys))[1:]:
			prev_key = self.black_keys[i - 1]
			distances.append(key.x - prev_key.x)
		mean_dist, std_dev = mean_and_standard_dev(distances)

		grouped_black_keys = [[self.black_keys[0]]]
		for i, key in tuple(enumerate(self.black_keys))[1:]:
			prev_key = self.black_keys[i - 1]
			if key.x - prev_key.x < mean_dist:
				grouped_black_keys[-1].append(key)
			else:
				grouped_black_keys.append([key])

		for group in grouped_black_keys:
			size = len(group)
			if size == 2:
				group[0].note = Note.C_SHARP
				group[1].note = Note.D_SHARP
			elif size == 3:
				group[0].note = Note.F_SHARP
				group[1].note = Note.G_SHARP
				group[2].note = Note.A_SHARP

		for i, group in enumerate(grouped_black_keys):
			if len(group) == 1:
				key = group[0]
				if i > 0 and (grouped_black_keys[i - 1][-1].note is not None):
					key.note = Note((grouped_black_keys[i - 1][-1].note.value + 2) % 7)
				elif i < len(grouped_black_keys) - 1 and (grouped_black_keys[i + 1][0].note is not None):
					key.note = Note((grouped_black_keys[i + 1][0].note.value - 2) % 7)

		self.white_keys.sort(key=lambda k: k.x)

		a1_index, a1 = None, None
		for i, key in enumerate(self.white_keys):
			for j in range(len(self.black_keys) - 1):
				black_left = self.black_keys[j]
				black_right = self.black_keys[j + 1]
				if black_left.note == Note.G_SHARP and black_right.note == Note.A_SHARP and black_left.x < key.x < black_right.x:
					key.note = Note.A
					key.octave = 1
					a1_index, a1 = i, key
					break
			if a1_index:
				break

		if a1:
			for i, key in enumerate(self.white_keys[a1_index + 1:]):
				dist = i + 1
				key.note = Note((Note.A.value + dist) % 7)
				key.octave = 1 + dist // 7

			for i, key in enumerate(self.white_keys[a1_index - 1::-1]):
				dist = i + 1
				key.note = Note((Note.A.value - dist) % 7)
				key.octave = -(dist // 7)

			a_sharp_1 = min(filter(lambda k: k.x > a1.x, self.black_keys), key=lambda k: k.x - a1.x)
			a_sharp_1.octave = 1
			a_sharp_1_index = self.black_keys.index(a_sharp_1)

			for i, key in tuple(enumerate(self.black_keys))[a_sharp_1_index + 1:]:
				dist = i - a_sharp_1_index
				key.octave = 1 + dist // 5

			for i, key in tuple(enumerate(self.black_keys))[a_sharp_1_index - 1::-1]:
				dist = a_sharp_1_index - i
				key.octave = -(dist // 5)
