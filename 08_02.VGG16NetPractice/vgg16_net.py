#coding: utf-8

#This file is the sample code of lecture vgg16.py
#Read the vgg16-net model file, reproduce the VGG16 CNN.

import os
import time
import inspect
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


POOLING_SIZE = 2
POOLING_STEP = 2

#Print necessary debug message with DEBUG = True.
DEBUG = False

#TODO:?
VGG_MEAN = [103.939, 116.799, 123.68]

class Vgg16(object):
	"""docstring for Vgg16"""
	def __init__(self, path=None):
		if path is None:
			path = os.path.join(os.getcwd(), "./vgg16.npy")

		self.data_dict = np.load(path, encoding='latin1').item()
		#if DEBUG: print self.data_dict


	def fw_propagation(self, images):
		
		if DEBUG:
			print "Build model started:"

		start = time.time()
		#vgg16网络直接输入图像原始像素值，不需要转化
		#rgb_scaled = images * 255.0
		rgb_scaled = images
		#if DEBUG: print rgb_scaled
		red, green, blue = tf.split(rgb_scaled, 3, 3)
		bgr = tf.concat([blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]], 3)

		#Convolution layher
		self.conv1_1 = self.conv_layer(bgr, "conv1_1")
		self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
		self.pool1   = self.max_pool_2X2(self.conv1_2, "pool1")
		

		self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
		self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
		self.pool2   = self.max_pool_2X2(self.conv2_2, "pool2")


		self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
		self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
		self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
		self.pool3   = self.max_pool_2X2(self.conv3_3, "pool3")


		self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
		self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
		self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
		self.pool4   = self.max_pool_2X2(self.conv4_3, "pool4")


		self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
		self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
		self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
		self.pool5   = self.max_pool_2X2(self.conv5_3, "pool5")


		#Full-connect layer
		self.fc6     = self.fc_layer(self.pool5, "fc6")
		self.relu6   = tf.nn.relu(self.fc6)

		self.fc7     = self.fc_layer(self.relu6, "fc7")
		self.relu7   = tf.nn.relu(self.fc7)

		self.fc8     = self.fc_layer(self.relu7, "fc8")
		self.prob    = tf.nn.softmax(self.fc8, name="prob")

		end = time.time()
		if DEBUG:
			print "Build model finished. Building takes %f." %(end - start)

		#why?
		self.data_dict = None

	def conv_layer(self, x, name):
		with tf.variable_scope(name):
			w = self.get_conv_filter(name)
			conv = tf.nn.conv2d(x, w, [1, 1, 1, 1], padding='SAME')
			conv_biases = self.get_bias(name)
			result = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))

			return result

	def get_conv_filter(self, name):
		return tf.constant(self.data_dict[name][0], name='filter')

	def get_bias(self, name):
		return tf.constant(self.data_dict[name][1], name='biases')

	def max_pool_2X2(self, x, name):
		return tf.nn.max_pool(x, ksize=[1,POOLING_SIZE,POOLING_SIZE,1],
								strides=[1,POOLING_STEP,POOLING_STEP,1],
								padding='SAME', name=name)

	def fc_layer(self, x, name):
		with tf.variable_scope(name):
			shape = x.get_shape().as_list()
			if DEBUG: print name, x.get_shape()
			dim = 1
			for i in shape[1:]:
				dim*=i
			
			x = tf.reshape(x, [-1, dim])
			w = self.get_fc_weight(name)
			b = self.get_bias(name)

			result = tf.nn.bias_add(tf.matmul(x, w), b)
			return result

	def get_fc_weight(self, name):
		return tf.constant(self.data_dict[name][0], name='weights')
