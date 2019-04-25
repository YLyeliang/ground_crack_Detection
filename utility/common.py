import tensorflow as tf
slim = tf.contrib.slim

@tf.contrib.framework.add_arg_scope
def conv2d(inputs,filters,kernel_size,strides=1):
    inputs = slim.conv2d(inputs,filters,kernel_size,stride=strides,padding='SAME')
    return inputs

# x=tf.placeholder(dtype=tf.float32,shape=(1,256,256,3))
# b=conv2d(x,32,3,2)
# print(b)
