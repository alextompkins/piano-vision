from copy import deepcopy
import cv2
from .processors import BlackKeyFinder, PianoEdgeFinder, ChangeTracker, HandFinder
from .video_reader import VideoReader


processors = [
	BlackKeyFinder(),
	PianoEdgeFinder(),
	ChangeTracker(),
	HandFinder()
]


def main():
	with VideoReader('data/canon_in_d-01.mp4') as video_reader:
		frame = video_reader.read_frame()

		while frame is not None:
			cv2.imshow('frame', frame)

			for processor in processors:
				processor.process_frame(deepcopy(frame))

			if cv2.waitKey(30) & 0xFF == ord('q'):
				break
			frame = video_reader.read_frame()


if __name__ == '__main__':
	main()
