#coding: utf-8

#This file is the sample code of lecture mnist_lenet5_foward.py

import tensorflow as tf

#All the 'cifar-10' image is 28 * 28 resolution
#0-9
OUTPUT_NODE = 10
LAYER1_NODE = 120
LAYER2_NODE = 86

#CNN Modification
IMAGE_XY_RES = 32
IMAGE_CHANS  = 3

CONV1_SIZE = 5
CONV1_KNELS = 32

CONV2_SIZE = 5
CONV2_KNELS = 16

POOLING_SIZE = 2
POOLING_STEP = 2

def get_weight(shape, regularizer):
	#w = tf.Variable(tf.random_normal(shape), dtype = tf.float32)
	w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
	if regularizer != None:
		tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

def get_bias(shape):
	#b = tf.Variable(tf.constant(0.01, shape = shape))
	b = tf.Variable(tf.zeros(shape))
	return b

def conv2d(x, w):
	return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def max_pool(x):
	return tf.nn.max_pool(x, ksize=[1,POOLING_SIZE,POOLING_SIZE,1],
		strides=[1,POOLING_STEP,POOLING_STEP,1], padding='SAME')

def fw_propagation(x, train, regularizer):

	#生成第一步卷积核及其每个核的偏置（w其实就是卷积核矩阵中的各元素）
	conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, IMAGE_CHANS, CONV1_KNELS], regularizer)
	conv1_b = get_bias([CONV1_KNELS])

	#执行卷积操作
	conv1 = conv2d(x, conv1_w)
	#激活与池化
	relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
	pool1 = max_pool(relu1)


	#生成第二步卷积核及其每个核的偏置（w其实就是卷积核矩阵中的各元素）
	conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KNELS, CONV2_KNELS], regularizer)
	conv2_b = get_bias([CONV2_KNELS])

	#执行卷积操作
	conv2 = conv2d(pool1, conv2_w)
	#激活与池化
	relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
	pool2 = max_pool(relu2)

	#对第二步操作的输出进行拉直操作
	pool_shape = pool2.get_shape().as_list()
	#pool shape is [200, 7, 7, 64]
	nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
	reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

	w1 = get_weight([nodes, LAYER1_NODE], regularizer)
	b1 = get_bias([LAYER1_NODE])
	y1 = tf.nn.relu(tf.matmul(reshaped, w1) + b1)

	w2 = get_weight([LAYER1_NODE, LAYER2_NODE], regularizer)
	b2 = get_bias([LAYER2_NODE])
	y2 = tf.nn.relu(tf.matmul(y1, w2) + b2)

	#执行dropout
	if train: y2 = tf.nn.dropout(y2, 0.5)

	w3 = get_weight([LAYER2_NODE, OUTPUT_NODE], regularizer)
	b3 = get_bias([OUTPUT_NODE])
	y = tf.matmul(y2, w3) + b3

	return y
