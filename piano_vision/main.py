from pathlib import Path

import cv2
from .processors import BlackKeyFinder, KeyboardBounder, ChangeTracker, HandFinder
from .video_reader import VideoReader


processors = [
	BlackKeyFinder(),
	ChangeTracker(),
	HandFinder()
]


class PianoVision:
	def __init__(self, video_name):
		self.video_file = 'data/{}.mp4'.format(video_name)
		self.ref_frame_file = 'data/{}-f00.png'.format(video_name)

		self.bounder = KeyboardBounder()
		self.bounds = [0, 0, 0, 0]

	def main_loop(self):
		with VideoReader(self.video_file) as video_reader:
			frame = video_reader.read_frame()

			# Use initial frame file if it exists, otherwise just use first frame
			if Path(self.ref_frame_file).exists():
				self.handle_reference_frame(cv2.imread(self.ref_frame_file))
			else:
				self.handle_reference_frame(frame)

			# Loop through remaining frames
			while frame is not None:
				keyboard = self.bounder.get_bounded_section(frame, self.bounds)
				cv2.imshow('keyboard', keyboard)

				cv2.rectangle(frame, self.bounds[0], self.bounds[3], (0, 255, 255), thickness=2)
				cv2.imshow('frame', frame)

				for processor in processors:
					processor.process_frame(keyboard.copy())

				if cv2.waitKey(30) & 0xFF == ord('q'):
					break
				frame = video_reader.read_frame()

	def handle_reference_frame(self, reference_frame):
		self.bounds = self.bounder.find_bounds(reference_frame)
