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

DEBUG = False
SHOW_DIAGRAM = False

def application(img_path):
	img_ready = utils.load_image(img_path)

	with tf.Session() as sess:
		images = tf.placeholder(tf.float32, [1, IMAGE_XY_RES, IMAGE_XY_RES, IMAGE_CHANS])
		vgg = vgg16.Vgg16("./model/vgg16.npy")
		vgg.fw_propagation(images)
		probability = sess.run(vgg.prob, feed_dict={images:img_ready})
		top5 = np.argsort(probability[0])[-1:-6:-1]
		if DEBUG: print "Top 5:", top5

		values = []
		bar_label = []
		for n, i in enumerate(top5):
			values.append(probability[0][i])
			bar_label.append(labels[i])
			if DEBUG:
				print i, ":", labels[i], "----", utils.percent(probability[0][i])

		if SHOW_DIAGRAM:
			fig = plt.figure(u"Top-5 预测结果")

			ax = fig.add_subplot(111)
			ax.bar(range(len(values)), values, tick_label=bar_label, width=0.5, fc='cyan')
			ax.set_ylabel(u"probability")
			ax.set_title(u"Top-5")

			for a, b in zip(range(len(values)), values):
				ax.text(a, b+0.005, utils.percent(b), ha='center', va='bottom', fontsize=7)

			plt.show()

	return bar_label[0], values[0]


def main():
	#while True:
	#	img_path = raw_input("Please inut the image path, 'Q' or 'q' for quit:\n")
	#	if img_path.strip() == 'Q' or img_path.strip() == 'q': break

	for i in xrange(0, 10):
		img_path = "./pic/%d.jpg" %i

		print "Start recognizing %s." %img_path
		res, prob = application(img_path)
		print "Picture %d.jpg has a %s probability of being a %s." %(i, utils.percent(prob), res)
		#reset for free memory, avoid causing std::alloc exception.
		tf.reset_default_graph()
	
	print "Quit the application."
	

if __name__ == '__main__':
	main()