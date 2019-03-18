#coding: utf-8

#This file is the sample code of lecture mnist_foward.py

import tensorflow as tf

#All the 'digit' image is 28 * 28 resolution
INPUT_NODE =  784
#0-9
OUTPUT_NODE = 10
LAYER1_NODE = 500


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

def fw_propagation(x, regularizer):

	w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
	b1 = get_bias([LAYER1_NODE])
	y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

	w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
	b2 = get_bias([OUTPUT_NODE])
	y = tf.matmul(y1, w2) + b2

	return y
