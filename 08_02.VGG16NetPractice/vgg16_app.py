#coding: utf-8

#This file is the sample code of lecture app.py
#Application file: read the file and implement recongnition.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import vgg16_net as vgg16
import vgg16_utils as utils
from vgg16_nclasses import labels


IMAGE_XY_RES = 224 
IMAGE_CHANS  = 3

def application():
	#img_path  = raw_input("Please inut the image path:")
	#img_ready = utils.load_image(img_path)
	img_ready = utils.load_image("./pic/a.jpg")

	fig = plt.figure(u"Top-5 预测结果")

	with tf.Session() as sess:
		images = tf.placeholder(tf.float32, [1, IMAGE_XY_RES, IMAGE_XY_RES, IMAGE_CHANS])
		vgg = vgg16.Vgg16("./model/vgg16.npy")
		vgg.fw_propagation(images)
		probability = sess.run(vgg.prob, feed_dict={images:img_ready})
		top5 = np.argsort(probability[0])[-1:-6:-1]
		print "Top 5:", top5

		values = []
		bar_label = []
		for n, i in enumerate(top5):
			print "n:", n
			print "i:", i
			values.append(probability[0][i])
			bar_label.append(labels[i])
			print i, ":", labels[i], "----", utils.percent(probability[0][i])

		ax = fig.add_subplot(111)
		ax.bar(range(len(values)), values, tick_label=bar_label, width=0.5, fc='g')
		ax.set_ylabel(u"probability")
		ax.set_title(u"Top-5")

		for a, b in zip(range(len(values)), values):
			ax.text(a, b+0.005, utils.percent(b), ha='center', va='bottom', fontsize=7)

		plt.show()


def main():
	application()

if __name__ == '__main__':
	main()