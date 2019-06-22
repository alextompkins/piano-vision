import cv2
import numpy as np
import time
from math import inf


def rotate_image(image, angle):
	image_center = tuple(np.array(image.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
	return result


def apply_mask(frame, mask):
	"""Apply binary mask to frame, return masked image."""
	return cv2.bitwise_and(frame, frame, mask=mask)


def mean_and_standard_dev(values, key=None):
	if key:
		values = tuple(map(key, values))

	return np.mean(values), np.std(values)


def dist(p1, p2):
	"""Calculates the Euclidean distance between two 2D points"""
	dx = p1[0][0] - p2[0][0]
	dy = p1[0][1] - p2[0][1]
	return np.sqrt(dx ** 2 + dy ** 2)


def group(data, radius, dist_func=dist):
	"""Clusters data that fall within radius of each by measure of dist_func"""
	clustered = [[data[0]]]
	for val in data[1:]:
		if dist_func(val, clustered[-1][0]) < radius:
			clustered[-1].append(val)
		else:
			clustered.append([val])

	return clustered


def avg_of_groups(point_groups):
	"""Given a list of groups of 2D points, returns a list with the averages
	of each group in the same order"""
	point_avgs = []

	for group in point_groups:
		point_avg = [[0, 0]]

		for point in group:
			point_avg[0][0] += point[0][0]
			point_avg[0][1] += point[0][1]

		point_avg[0][0] = int(round(point_avg[0][0] / len(group)))
		point_avg[0][1] = int(round(point_avg[0][1] / len(group)))

		point_avgs += [point_avg]

	return point_avgs


def index_of_closest(data, points, dist_func=dist):
	"""Generates an index of the closest values in data to values in points by
	measure of dist_func"""
	ioc = []

	for point in points:
		dist_curr = float(inf)
		indx = -1
		for i, dpoint in enumerate(data):
			disti = dist_func(point, dpoint)
			if disti < dist_curr:
				dist_curr = disti
				indx = i
		ioc += [indx]

	return ioc


def centre_of_contour(contour):
	moments = cv2.moments(contour)
	centre_x = int(moments['m10'] / moments['m00'])
	centre_y = int(moments['m01'] / moments['m00'])
	return centre_x, centre_y


def epochtime_ms():
	return int(round(time.time() * 1000))
