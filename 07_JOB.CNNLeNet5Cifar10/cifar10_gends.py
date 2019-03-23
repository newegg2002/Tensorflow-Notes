#coding: utf-8

#This file is the sample code of lecture mnist_generateds.py

import tensorflow as tf
import numpy as np
from PIL import Image
import os

IMG_TRAIN_PATH = "./cifar10_data/train/"
IMG_TEST_PATH  = "./cifar10_data/test/"

TFR_PATH = "./tfrecords"
TFRECORD_TRAIN = os.path.join(TFR_PATH, "cifar10_train.tfr")
TFRECORD_TEST  = os.path.join(TFR_PATH, "cifar10_test.tfr")

CIFAR_CAT2IDX_LUT = {"airplane":0, "automobile":1, "bird":2, "cat":3, "deer":4, 
                     "dog":5, "frog":6, "horse":7, "ship":8, "truck":9}

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
	img.set_shape([3072])
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

	#cifar10_data的目录结构为：[test, train]/[airplane, ship, ..., dog]/[batch1_num_xxx.jpg, ...]

	#遍历当前目录，获取category文件夹列表
	cat_dirs = os.listdir(img_path)

	#针对每个category目录进行遍历，获取所有jpg文件列表。注意因为我们是基于img_path进行访问，所以
	#文件路径中要加上category。
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

	writer = tf.python_io.TFRecordWriter(tfR_path)

	pic_cnt = 0
	for (f_img, label) in zip(img_files_list, labels_list):
		img_name = os.path.join(img_path, f_img)
		img = Image.open(img_name)
		img_raw = img.tobytes()

		labels = [0] * 10
		labels[CIFAR_CAT2IDX_LUT[label]] = 1

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