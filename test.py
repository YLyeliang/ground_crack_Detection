# a python file used to do some function test.
#
#
import tensorflow as tf
import numpy as np

sess =tf.Session()
H,W =5,10

b=[]
a=np.arange(10).reshape([10,1])
c= a.sum(axis=-1)>5

a=[1,2,3,4]
for i in range(3):
    b += np.argmax(a, axis=-1).tolist()
c =np.concatenate(b,axis=1)





a=np.arange(10).reshape([5,1,2])
argmax = np.argmax(a,axis=-1)
wh=np.expand_dims(a,axis=-2)
boxes_max = wh / 2.
boxes_min = -boxes_max
anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
b= anchor_mask[1].index(3)

a = tf.range(10)
a =tf.reshape(a,[5,2])
b,c,d,e =tf.unstack(a,4,axis=0)


f= tf.stack([b,c,d,e],axis=1)
# a=tf.range(100)
# a=tf.reshape(a,[1,2,2,25])
# centers,sizes,conf,prob =tf.split(a,[2,2,1,20],axis=-1)
# print(sess.run(a))
# print(b)

x=tf.range(W)
y=tf.range(H)
a,b=tf.meshgrid(x,y)
x_offset = tf.reshape(a,(-1,1))
y_offset = tf.reshape(b,(-1,1))
x_y_offset = tf.concat([x_offset,y_offset],axis=-1)
x_y_offset = tf.reshape(x_y_offset,[H,W,1,2])
print(x_y_offset)
c,d=sess.run([a,b])
offx,offy,xy=sess.run([x_offset,y_offset,x_y_offset])
print(a,b)
print(c)
print(d)
print(offx)
print(offy  )
print(xy)
