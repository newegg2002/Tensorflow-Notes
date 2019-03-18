#coding: utf-8

#This file is the sample code of lecture mnist_generateds.py

import tensorflow as tf
import numpy as np
from PIL import Image
import os

img_train_path = "./mnist_data_jpg/mnist_train_jpg_60000/"
label_train_path = "./mnist_data_jpg/mnist_train_jpg_60000.txt"
tfRecord_train = "./data/mnist_train.tfrecords"

img_test_path = "./mnist_data_jpg/mnist_test_jpg_10000/"
label_test_path = "./mnist_data_jpg/mnist_test_jpg_10000.txt"
tfRecord_test = "./data/mnist_test.tfrecords"

data_path = "./data"

resize_height = 28
resize_width  = 28


def read_tfRecord(tfR_path):
	filename_q = tf.train.string_input_producer([tfR_path])

	reader = tf.TFRecordReader()
	_, serialize_example = reader.read(filename_q)

	features = tf.parse_single_example(serialize_example,
		features = {
		'label':   tf.FixedLenFeature([10], tf.int64),
		'img_raw': tf.FixedLenFeature([],   tf.string)
		})

	img = tf.decode_raw(features['img_raw'], tf.uint8)
	img.set_shape([784])
	img = tf.cast(img, tf.float32) * (1. / 255)

	label = tf.cast(features['label'], tf.float32)

	return img, label


def get_tfRecords(num, isTrain=True):
	if isTrain:
		tfR_path = tfRecord_train
	else:
		tfR_path = tfRecord_test

	img, label = read_tfRecord(tfR_path)
	img_batch, label_batch = tf.train.shuffle_batch(
		[img, label],
		batch_size = num,
		num_threads = 2,
		capacity = 10000,
		min_after_dequeue = 700)

	return img_batch, label_batch



def write_tfRecord(tfR_path, img_path, label_path):
	writer = tf.python_io.TFRecordWriter(tfR_path)

    #Get all labels data
	fLabel = open(label_path)
	labels_data = fLabel.readlines()
	fLabel.close()

	pic_cnt = 0
	for l in labels_data:
		val = l.split()

		img_name = img_path + val[0]
		img = Image.open(img_name)
		img_raw = img.tobytes()

		labels = [0] * 10
		labels[int(val[1])] = 1

		example = tf.train.Example(
			features=tf.train.Features(
				feature={
				'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
				'label':  tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
				}))
		writer.write(example.SerializeToString())
		pic_cnt += 1
		print "%d pics processed.\r" %(pic_cnt),

	writer.close()
	print "Write tfRecords sucessful"


def gen_tfRecords():
	isExists = os.path.exists(data_path)

	if not isExists:
		os.mkdirs(data_path)
	print "tfRecord data would be put in " + data_path

	write_tfRecord(tfRecord_train, img_train_path, label_train_path)
	write_tfRecord(tfRecord_test,  img_test_path,  label_test_path)

def main():
	gen_tfRecords()
	#_, _ = get_tfRecords(500)
	#_, _ = get_tfRecords(1000, False)


if __name__ == "__main__":
	main()