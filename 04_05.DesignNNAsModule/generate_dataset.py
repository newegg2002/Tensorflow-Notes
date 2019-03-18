#coding: utf-8

#This file is the sample code of lecture opt4_8_generateds.py

import numpy as np

RSEED = 2
INPUT_SIZE = 300

def generateds():
	rdm = np.random.RandomState(RSEED)
	X = rdm.randn(300, 2)

	Y_ = [int(x0 * x0 + x1 * x1 < 2) for (x0, x1) in X]
	Y_c = ["r" if y_ else "b" for y_ in Y_]

	X = np.vstack(X).reshape(-1, 2)
	Y_ = np.vstack(Y_).reshape(-1, 1)

	return X, Y_, Y_c