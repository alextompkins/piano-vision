from pathlib import Path

import cv2
import numpy as np

from .processors import KeysManager, KeyboardBounder, HandFinder, PressedKeyDetector
from .video_reader import VideoReader


class PianoVision:
	def __init__(self, video_name):
		self.video_file = 'data/{}.mp4'.format(video_name)
		self.ref_frame_file = 'data/{}-f00.png'.format(video_name)

		self.reference_frame = None

		self.bounder = KeyboardBounder()
		self.bounds = [0, 0, 0, 0]

		self.hand_finder = HandFinder()
		self.keys_manager = None
		self.pressed_key_detector = None

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

				# Use morphological closing to join up hand segments
				# TODO maybe replace this with joining nearby contours?
				kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
				skin_mask_closed = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
				# cv2.imshow('skin_mask_closed', skin_mask_closed)
				hand_contours = self.hand_finder.get_hand_contours(skin_mask_closed)

				fingertips = self.hand_finder.find_fingertips(hand_contours)
				flat_fingertips = []
				for hand in fingertips:
					flat_fingertips.extend(hand)

				pressed_keys = self.pressed_key_detector.detect_pressed_keys(keyboard, skin_mask, flat_fingertips)

				cv2.imshow('keyboard vs. ref', np.vstack([keyboard, self.reference_frame]))

				# Show frame with keys overlaid
				for key in self.keys_manager.white_keys:
					x, y, w, h = key.x, key.y, key.width, key.height
					cv2.rectangle(keyboard, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=key in pressed_keys and cv2.FILLED or 1)
					cv2.putText(keyboard, key.note.pretty_name(), (x + 3, y + h - 10), cv2.FONT_HERSHEY_PLAIN, 0.75, color=(0, 0, 255))
				for key in self.keys_manager.black_keys:
					x, y, w, h = key.x, key.y, key.width, key.height
					cv2.rectangle(keyboard, (x, y), (x + w, y + h), color=(255, 150, 75), thickness=key in pressed_keys and cv2.FILLED or 1)
					cv2.putText(keyboard, key.note.pretty_name(), (x, y + h - 10), cv2.FONT_HERSHEY_PLAIN, 0.75, color=(255, 150, 75))

				if hand_contours:
					cv2.drawContours(keyboard, tuple(hand_contours), -1, color=(0, 255, 0), thickness=1)

				# Highlight detected fingertips
				for hand in fingertips:
					for finger in hand:
						if finger:
							cv2.circle(keyboard, finger, radius=5, color=(0, 255, 0), thickness=2)

				cv2.imshow('keyboard', keyboard)

				# Wait for 30ms then get next frame unless quit
				pressed_key = cv2.waitKey(30) & 0xFF
				if pressed_key == 32:  # spacebar
					paused = not paused
				elif pressed_key == ord('r'):
					self.handle_reference_frame(frame)
				elif pressed_key == ord('q'):
					break
				if not paused:
					frame = video_reader.read_frame()

	def handle_reference_frame(self, reference_frame):
		self.bounds = self.bounder.find_bounds(reference_frame)
		self.reference_frame = self.bounder.get_bounded_section(reference_frame, self.bounds)
		self.keys_manager = KeysManager(self.reference_frame)
		self.pressed_key_detector = PressedKeyDetector(self.reference_frame, self.keys_manager)

		print('{} black keys found'.format(len(self.keys_manager.black_keys)))
		print('{} white keys found'.format(len(self.keys_manager.white_keys)))
