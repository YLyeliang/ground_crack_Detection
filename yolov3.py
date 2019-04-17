import tensorflow as tf
import numpy as np
import cv2
import os

a=np.arange(24).reshape([2,2,3,2])
print(a.shape)
b=a[3:4]
c=a[::-1]
print(c)
print(b)