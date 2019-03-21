#coding: utf-8

#This file is the sample code of lecture mnist_generateds.py

import tensorflow as tf
import numpy as np
from PIL import Image
import os

IMG_TRAIN_PATH = "./cifar10_data/train/"
IMG_TEST_PATH  = "./cifar10_data/test/"

TFR_PATH = "./tfrecords"
TFRECORD_TRAIN = "cifar10_train.tfr"
TFRECORD_TEST  = "cifar10_test.tfr"

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
		tfR_path = TFRECORD_TRAIN
	else:
		tfR_path = TFRECORD_TEST

	img, label = read_tfRecord(tfR_path)
	img_batch, label_batch = tf.train.shuffle_batch(
		[img, label],
		batch_size = num,
		num_threads = 2,
		capacity = 10000,
		min_after_dequeue = 700)

	return img_batch, label_batch


def get_img_files_and_labels_list(img_path):
	
	img_files_list = []
	labels_list = []

	cat_dirs = os.listdir(img_path)

	for cat in cat_dirs:
		cat_dir = os.path.join(img_path, cat)
		if os.path.isdir(cat_dir):
			files = os.listdir(cat_dir)
			img_files_list.extend(cat + "/" + file for file in files)
			labels_list.extend(len(files) * [cat])

	return img_files_list, labels_list

def write_tfRecord(tfR_path, img_path):

	#先遍历img_path，获取所有image和对应label的列表
	img_files_list, labels_list = get_img_files_and_labels_list(img_path)

	return

	#根据获取到的文件和标签列表，生成tfrecord并写入
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
	isExists = os.path.exists(TFR_PATH)

	if not isExists:
		os.mkdir(TFR_PATH)
	print "tfRecords data would be put in " + TFR_PATH

	write_tfRecord(TFRECORD_TRAIN, IMG_TRAIN_PATH)
	write_tfRecord(TFRECORD_TEST,  IMG_TEST_PATH)

def main():
	gen_tfRecords()
	#_, _ = get_tfRecords(500)
	#_, _ = get_tfRecords(1000, False)


if __name__ == "__main__":
	main()