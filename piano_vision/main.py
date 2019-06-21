from pathlib import Path

import cv2
import numpy as np

from .processors import KeysManager, KeyboardBounder, HandFinder
from .video_reader import VideoReader
from .helpers import apply_mask


class PianoVision:
	def __init__(self, video_name):
		self.video_file = 'data/{}.mp4'.format(video_name)
		self.ref_frame_file = 'data/{}-f00.png'.format(video_name)

		self.reference_frame = None

		self.bounder = KeyboardBounder()
		self.bounds = [0, 0, 0, 0]

		self.hand_finder = HandFinder()
		self.keys_manager = None

	def main_loop(self):
		with VideoReader(self.video_file) as video_reader:
			paused = False
			frame = video_reader.read_frame()

			# Use initial frame file if it exists, otherwise just use first frame
			if Path(self.ref_frame_file).exists():
				self.handle_reference_frame(cv2.imread(self.ref_frame_file))
			else:
				self.handle_reference_frame(frame)

			# Loop through remaining frames
			while frame is not None:
				keyboard = self.bounder.get_bounded_section(frame, self.bounds)

				cv2.rectangle(frame, self.bounds[0], self.bounds[3], (0, 255, 255), thickness=2)
				cv2.imshow('frame', frame)

				skin_mask = self.hand_finder.get_skin_mask(keyboard)
				# Dilate again to ensure that we don't include any small bits of skin
				kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
				dilated_mask = cv2.dilate(skin_mask, kernel, iterations=1)

				skin = apply_mask(keyboard, dilated_mask)
				keyboard = cv2.subtract(keyboard, skin)

				skin_ref = apply_mask(self.reference_frame, dilated_mask)
				ref = cv2.subtract(self.reference_frame, skin_ref)

				diff = cv2.absdiff(keyboard, ref)
				diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
				# diff = cv2.GaussianBlur(diff, (3, 3), 0)
				diff = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 10)
				kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
				diff = cv2.dilate(diff, kernel, iterations=2)
				diff = cv2.erode(diff, kernel, iterations=2)

				cv2.imshow('keyboard vs. ref', np.vstack([keyboard, ref]))
				cv2.imshow('diff', diff)

				# Show frame with keys overlaid
				for rect in self.keys_manager.black_keys:
					x, y, w, h = rect
					cv2.rectangle(keyboard, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=1)
				for rect in self.keys_manager.white_keys:
					x, y, w, h = rect
					cv2.rectangle(keyboard, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=1)
				cv2.imshow('keyboard', keyboard)

				# Wait for 30ms then get next frame unless quit
				pressed_key = cv2.waitKey(30) & 0xFF
				if pressed_key == 32:  # spacebar
					paused = not paused
				elif pressed_key == ord('q'):
					break
				if not paused:
					frame = video_reader.read_frame()

	def handle_reference_frame(self, reference_frame):
		self.bounds = self.bounder.find_bounds(reference_frame)
		self.reference_frame = self.bounder.get_bounded_section(reference_frame, self.bounds)
		self.keys_manager = KeysManager(self.reference_frame)

		print('{} black keys found'.format(len(self.keys_manager.black_keys)))
		print('{} white keys found'.format(len(self.keys_manager.white_keys)))
