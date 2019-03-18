#encoding: utf-8

import turtle

def drawhouse():

	t = turtle.Pen()

	#H-> default
	t.right(30)
	t.forward(223)

	t.left(30)
	t.backward(80)

	t.right(90)
	t.forward(200)

	t.right(90)
	t.forward(240)

	t.right(90)
	t.forward(200)

	t.left(90)
	t.forward(80)

	t.right(152)
	t.forward(240)


	#reset() to clean and reset
	#t.reset()


drawhouse()
c = raw_input()