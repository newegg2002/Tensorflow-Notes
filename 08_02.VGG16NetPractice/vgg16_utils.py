#coding: utf-8

#This file is the sample code of lecture utils.py
#Read the input picture, show the probability diagram.

from skimage import io, transform, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pylab import mpl

import vgg16_app as app

DEBUG = False
SHOW_DIAGRAM = False


#正常显示中文标签
mpl.rcParams['font.sans-serif'] = ['SimHei']
#正常显示正负号
mpl.rcParams['axes.unicode_minus'] = False

def load_image(path):
	img = io.imread(path)

	#vgg16网络直接输入图像原始像素值，不需要转化
	#img = img / 255.0

	if SHOW_DIAGRAM:
		fig = plt.figure("Centre and Resize:")

		ax0 = fig.add_subplot(131)
		ax0.set_xlabel(u'Original Picture')
		ax0.imshow(img)

	short_edge = min(img.shape[:2])
	y = (img.shape[0] - short_edge ) / 2
	x = (img.shape[1] - short_edge ) / 2
	crop_img = img[y:y+short_edge, x:x+short_edge]

	if DEBUG: 
		print short_edge, x, y
		print img.shape, crop_img.shape

	if SHOW_DIAGRAM:
		ax1 = fig.add_subplot(132)
		ax1.set_xlabel(u'Centre Picture')
		ax1.imshow(crop_img)

	#使用preserve_range参数避免resize时对像素值范围作转化。
	re_img = transform.resize(crop_img, (app.IMAGE_XY_RES, app.IMAGE_XY_RES), preserve_range=True)

	if SHOW_DIAGRAM:
		ax2 = fig.add_subplot(133)
		ax2.set_xlabel(u'Resize Picture')
		#imshow内部的参数类型可以分为两种:
		#（1）当输入矩阵是uint8类型的时候，此时imshow显示图像的时候，会认为输入矩阵的值范围在0-255之间;
		#（2）如果imshow的参数是double类型的时候，那么imshow会认为输入矩阵值的范围在0-1
		ax2.imshow(re_img.astype(np.uint8))

	img_ready = re_img.reshape(1, app.IMAGE_XY_RES, app.IMAGE_XY_RES, app.IMAGE_CHANS)

	if SHOW_DIAGRAM:
		plt.show()

	return img_ready

def percent(value):
	return "%.2f%%" %(value * 100)