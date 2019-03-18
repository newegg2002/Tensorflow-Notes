#encoding: utf-8

import tensorflow as tf

#scalar
sa = tf.constant(1.0)
sb = tf.constant(2.0)

ssum = sa + sb
print "Graph of Scalar(a+b) is:", ssum

#vector
va = tf.constant([1.0, 2.0])
vb = tf.constant([3.0, 4.0])

vsum = va + vb
print "Graph of Vector(a+b) is:", vsum

#matrix
ma = tf.constant([[1, 2], [3, 4], [5, 6]])
mb = tf.constant([[7, 8], [0, 9], [2, 8]])

msum = ma + mb
print "Graph of Matrix(a+b) is:", msum

#tensor, n = 4
ta = tf.constant([[[[10, 11, 12], [20, 21, 22]],
                   [[30, 31, 32], [40, 41, 42]]
                  ],
                  [[[13, 14, 15], [23, 24, 25]],
                   [[13, 14, 15], [23, 24, 25]]]
                 ])
tb = tf.constant([[[[10, 11, 12], [20, 21, 22]],
                   [[30, 31, 32], [40, 41, 42]]
                  ],
                  [[[13, 14, 15], [23, 24, 25]],
                   [[13, 14, 15], [23, 24, 25]]]
                 ])

tsum = ta + tb
print "Graph of Tensor(a+b) is:", tsum
with tf.Session() as tsessAdd:
    print tsessAdd.run(tsum)

#multiply
mx = tf.constant([[1.0, 2.0]])
mw = tf.constant([[3.0], [4.0]])

mmulti = tf.matmul(mx, mw)
print "Graph of Matrix(a*b) is:", mmulti
with tf.Session() as msessMul:
    print msessMul.run(mmulti)