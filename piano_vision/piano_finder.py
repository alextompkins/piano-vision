import numpy as np
import cv2

from piano_vision.video_reader import VideoReader


class PianoFinder:
	pass


def main():
	with VideoReader('data/canon_in_d-01.mp4') as video_reader:
		bg_subtractor = cv2.createBackgroundSubtractorMOG2()
		frame = video_reader.read_frame()

		while frame is not None:
			# Translate to greyscale
			grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# Gaussian blur
			blur = cv2.GaussianBlur(grey, (3, 3), 0)

			# Threshold
			thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 99, 10)

			# Canny edge detection
			edges = cv2.Canny(grey, 100, 200)

			# Hough lines
			lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=100, maxLineGap=50)
			if lines is not None:
				for line in lines:
					for x1, y1, x2, y2 in line:
						cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

			# Apply background subtractor
			fg_mask = bg_subtractor.apply(grey)

			cv2.imshow('frame', frame)
			cv2.imshow('edges', edges)
			cv2.imshow('fg_mask', fg_mask)
			if cv2.waitKey(30) & 0xFF == ord('q'):
				break
			frame = video_reader.read_frame()


if __name__ == '__main__':
	main()
