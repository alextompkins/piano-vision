import cv2
import numpy as np


def rotate_image(image, angle):
	image_center = tuple(np.array(image.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
	return result


def apply_mask(frame, mask):
	"""Apply binary mask to frame, return masked image."""
	return cv2.bitwise_and(frame, frame, mask=mask)
