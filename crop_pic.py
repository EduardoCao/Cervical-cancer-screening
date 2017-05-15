import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import pandas as pd
import cv2
import math
from sklearn import mixture
from sklearn.utils import shuffle
from skimage import measure
from glob import glob
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from subprocess import check_output
from segmentation import get_filename
TRAIN_DATA = "./input/train"

TEST_DATA = "./input/test"

types = ['Type_1','Type_2','Type_3']
type_ids = []
type_ids_test = []

# Train set
for type in enumerate(types):
	type_i_files = glob(os.path.join(TRAIN_DATA, type[1], "*.jpg"))
	type_i_ids = np.array([s[len(TRAIN_DATA)+8:-4] for s in type_i_files])
	type_ids.append(type_i_ids)

# Test set
type_i_files = glob(os.path.join(TEST_DATA, "*.jpg"))
type_i_ids = np.array([s[len(TEST_DATA)+1:-4] for s in type_i_files])
type_ids_test.append(type_i_ids)

def get_image_data(image_id, image_type, isTest = False):
	"""
	Method to get image data as np.array specifying image id and type
	"""
	fname = get_filename(image_id, image_type, isTest)
	img = cv2.imread(fname)
	assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
	return img

def crop_pic(image_id, image_type, x, y, w, h, img_init_0, img_init_1, img_shp_0, img_shp_1, neww, newh):
	path = "./output_resize_%s_%s/" % (neww, newh) + image_type + "/"
	if not os.path.exists(path):
		os.makedirs(path)
	img = get_image_data(image_id, image_type)
	xx = int(float(x) / float(img_shp_1) * float(img_init_0))
	yy = int(float(y) / float(img_shp_0) * float(img_init_1))
	ww = int(float(w) / float(img_shp_1) * float(img_init_0))
	hh = int(float(h) / float(img_shp_0) * float(img_init_1))
	newimg = img[yy:yy+hh,xx:xx+ww]
	res = cv2.resize(newimg, (neww, newh))
	cv2.imwrite( path + image_id + ".jpg", res)

def crop_pic_test(image_id, image_type, x, y, w, h, img_init_0, img_init_1, img_shp_0, img_shp_1, neww, newh):
	path = "./output_resize_test_%s_%s/" % (neww, newh)
	if not os.path.exists(path):
		os.makedirs(path)
	img = get_image_data(image_id, image_type, isTest = True)
	xx = int(float(x) / float(img_shp_1) * float(img_init_0))
	yy = int(float(y) / float(img_shp_0) * float(img_init_1))
	ww = int(float(w) / float(img_shp_1) * float(img_init_0))
	hh = int(float(h) / float(img_shp_0) * float(img_init_1))
	newimg = img[yy:yy+hh,xx:xx+ww]
	res = cv2.resize(newimg, (neww, newh))
	cv2.imwrite(path + image_id + ".jpg", res)

if __name__ == '__main__':
	f = open("rectangles_train.csv", 'r')
	lines = f.readlines()
	for line in lines:
		if "image_id" in line:
			continue
		items = line.split(',')
		# crop_pic_test(items[0], items[1], items[2],items[3],items[4],items[5],items[6],items[7],items[8],items[9],64, 64)
		crop_pic(items[0], items[1], items[2],items[3],items[4],items[5],items[6],items[7],items[8],items[9],64, 64)