#coding: utf-8

#This file is the sample code of lecture utils.py
#Read the input picture, show the probability diagram.

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pylab import mpl

import vgg16_app as app


#正常显示中文标签
mpl.rcParams['font.sans-serif'] = ['SimHei']
#正常显示正负号
mpl.rcParams['axes.unicode_minus'] = False

def load_image(path):
	fig = plt.figure("Centre and Resize:")
	img = io.imread(path)
	img = img / 255.0

	ax0 = fig.add_subplot(131)
	ax0.set_xlabel(u'Original Picture')
	ax0.imshow(img)

	short_edge = min(img.shape[:2])
	y = (img.shape[0] - short_edge ) / 2
	x = (img.shape[1] - short_edge ) / 2
	crop_img = img[y:y+short_edge, x:x+short_edge]


	ax1 = fig.add_subplot(132)
	ax1.set_xlabel(u'Centre Picture')
	ax1.imshow(crop_img)

	re_img = transform.resize(crop_img, (app.IMAGE_XY_RES, app.IMAGE_XY_RES))


	ax2 = fig.add_subplot(133)
	ax2.set_xlabel(u'Resize Picture')
	ax2.imshow(re_img)

	img_ready = re_img.reshape(1, app.IMAGE_XY_RES, app.IMAGE_XY_RES, app.IMAGE_CHANS)

	return img_ready

def percent(value):
	return "%.2f%%" %(value * 100)