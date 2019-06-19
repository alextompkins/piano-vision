import cv2


class ChangeTracker:
	def __init__(self):
		self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()

	def process_frame(self, frame):
		# Apply background subtractor
		grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		fg_mask = self.bg_subtractor.apply(grey)
		cv2.imshow('fg_mask', fg_mask)
