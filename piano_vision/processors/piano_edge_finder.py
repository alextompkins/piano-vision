import cv2
import numpy as np


class PianoEdgeFinder:
	def process_frame(self, frame):
		grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(grey, 100, 200)
		lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=100, maxLineGap=50)
		if lines is not None:
			for line in lines:
				for x1, y1, x2, y2 in line:
					cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
		cv2.imshow('edges', frame)
