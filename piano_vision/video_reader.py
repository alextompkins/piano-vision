import cv2


class VideoReader:
	def __init__(self, video_file):
		self.video_file = video_file

	def __enter__(self):
		self.video = cv2.VideoCapture(self.video_file)
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.video.release()

	def read_frame(self):
		ret, frame = self.video.read()
		if ret:
			return frame
		else:
			return None
