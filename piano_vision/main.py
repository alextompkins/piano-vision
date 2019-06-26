from pathlib import Path

import cv2
import numpy as np

from .helpers import rotate_image
from .processors import KeysManager, KeyboardBounder, HandFinder, PressedKeyDetector
from .video_reader import VideoReader


class PianoVision:
	DELAY = 15  # delay between reading frames
	SNAPSHOT_INTERVAL = 30  # how many frames between snapshots, videos usually 30fps
	NUM_SNAPSHOTS = 20

	def __init__(self, video_name):
		self.video_name = video_name
		self.video_file = 'data/{}.mp4'.format(video_name)
		self.ref_frame_file = 'data/{}-f00.png'.format(video_name)

		self.reference_frame = None

		self.bounder = KeyboardBounder()
		self.bounds = [0, 0, 0, 0]

		self.hand_finder = HandFinder()
		self.keys_manager = None
		self.pressed_key_detector = None

		self.frame_counter = 0

	def main_loop(self):
		open('output/{}.log'.format(self.video_name), 'w').close()

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
				cv2.imshow('frame', frame)
				keyboard = self.bounder.get_bounded_section(frame, self.bounds)
				# cv2.imshow('post_warp', keyboard)

				skin_mask = self.hand_finder.get_skin_mask(keyboard)

				# Use morphological closing to join up hand segments
				# TODO maybe replace this with joining nearby contours?
				kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
				skin_mask_closed = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
				# cv2.imshow('skin_mask_closed', skin_mask_closed)
				hand_contours = self.hand_finder.get_hand_contours(skin_mask_closed)

				fingertips = self.hand_finder.find_fingertips(hand_contours, keyboard)
				flat_fingertips = []
				for hand in fingertips:
					flat_fingertips.extend(hand)

				pressed_keys = self.pressed_key_detector.detect_pressed_keys(keyboard, skin_mask, flat_fingertips)

				# cv2.imshow('keyboard vs. ref', np.vstack([keyboard, self.reference_frame]))

				# Show frame with keys overlaid
				for key in self.keys_manager.white_keys:
					x, y, w, h = key.x, key.y, key.width, key.height
					cv2.rectangle(keyboard, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=key in pressed_keys and cv2.FILLED or 1)
					cv2.putText(keyboard, str(key), (x + 3, y + h - 10), cv2.FONT_HERSHEY_PLAIN, 0.75, color=(0, 0, 255))
				for key in self.keys_manager.black_keys:
					x, y, w, h = key.x, key.y, key.width, key.height
					cv2.rectangle(keyboard, (x, y), (x + w, y + h), color=(255, 150, 75), thickness=key in pressed_keys and cv2.FILLED or 1)
					cv2.putText(keyboard, str(key), (x, y + h - 10), cv2.FONT_HERSHEY_PLAIN, 0.75, color=(255, 150, 75))

				if hand_contours:
					cv2.drawContours(keyboard, tuple(hand_contours), -1, color=(0, 255, 0), thickness=1)

				# Highlight detected fingertips
				for hand in fingertips:
					for finger in hand:
						if finger:
							cv2.circle(keyboard, finger, radius=5, color=(0, 255, 0), thickness=2)

				cv2.imshow('keyboard', keyboard)

				# Wait for 30ms then get next frame unless quit
				pressed_key = cv2.waitKey(self.DELAY) & 0xFF
				if pressed_key == 32:  # spacebar
					paused = not paused
				elif pressed_key == ord('r'):
					self.handle_reference_frame(frame)
				elif pressed_key == ord('q'):
					break
				if not paused:
					if self.frame_counter % self.SNAPSHOT_INTERVAL == 0:
						snapshot_index = self.frame_counter // self.SNAPSHOT_INTERVAL
						self.take_snapshot(snapshot_index, frame, keyboard, pressed_keys)
					self.frame_counter += 1
					frame = video_reader.read_frame()

	def handle_reference_frame(self, reference_frame):
		rotation = self.bounder.find_rotation(reference_frame)
		print('rotation: {}'.format(rotation))
		# reference_frame = rotate_image(reference_frame, rotation)

		self.bounds = self.bounder.find_bounds(reference_frame)
		self.reference_frame = self.bounder.get_bounded_section(reference_frame, self.bounds)
		self.keys_manager = KeysManager(self.reference_frame)
		self.pressed_key_detector = PressedKeyDetector(self.reference_frame, self.keys_manager)

		print('{} black keys found'.format(len(self.keys_manager.black_keys)))
		print('{} white keys found'.format(len(self.keys_manager.white_keys)))

	def take_snapshot(self, snapshot_index, frame, keyboard, pressed_keys):
		if snapshot_index < self.NUM_SNAPSHOTS:
			cv2.imwrite(
				'output/{}-snapshot{:02d}.png'.format(self.video_name, snapshot_index),
				np.vstack([frame, keyboard])
			)
			with open('output/{}.log'.format(self.video_name), 'a+') as log:
				line = '{}: [{}]\n'.format(snapshot_index, ', '.join([str(key) for key in pressed_keys]))
				log.write(line)
				print(line, end='')
