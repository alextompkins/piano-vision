import cv2


class BlackKeyFinder:
	def process_frame(self, frame):
		grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(grey, (3, 3), 0)
		thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 99, 10)
		cv2.imshow('thresholded', thresh)
