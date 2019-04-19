import colorsys
import numpy as np
import tensorflow as tf
from PIL import ImageFont,ImageDraw
from collections import Counter

def resize_image_correct_bbox(image,boxes,image_h,image_w):
    origin_image_size = tf.to_float(tf.shape(image)[0:2])
    image=tf.image.resize_images(image,size=[image_h,image_w])

    # correct bbox
    xx1 =boxes[:,0] *image_w/origin_image_size[1]
    yy1=boxes[:,1]*image_h/origin_image_size[0]
    xx2=boxes[:,2]*image_w/origin_image_size[1]
    yy2=boxes[:,3]*image_h/origin_image_size[0]
    idx= boxes[:,4]

    boxes=tf.stack([xx1,yy1,xx2,yy2,idx],axis=1)
    return image,boxes