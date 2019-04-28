import os
import tensorflow as tf
from utility import utils,yolov3
from data.dataset import dataset,Parser
from PIL import Image
import numpy as np

sess=tf.Session()

IMAGE_H, IMAGE_W = 416,416
CLASSES = {0:'crack'}
ANCHORS = utils.get_anchors('./data/anchors.txt',IMAGE_H,IMAGE_W)
NUM_CLASSES = len(CLASSES)
image_path= "/home/yel/data/Aerialgoaf/test"
temp = os.listdir(image_path)
files=[]
for x in temp:
    if 'xml' not in x:
       files+=[x]
def test():
    images =tf.placeholder(tf.float32,(1,416,416,3))
    model = yolov3.yolov3(NUM_CLASSES, ANCHORS)
    with tf.variable_scope("yolov3"):
        pred_feature_map = model.inference(images, is_training=False)
        y_pred = model.predict(pred_feature_map)

    # sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    saver = tf.train.Saver()
    saver.restore(sess,save_path="/home/yel/PycharmProjects/lab/model.ckpt-19000")

    for image in files:
        image =Image.open(os.path.join(image_path,image))
        image_resized = np.array(image.resize(size=(IMAGE_W, IMAGE_H)), dtype=np.float32)
        image_resized = image_resized/255.
        run_items = sess.run([ y_pred],feed_dict={images:np.expand_dims(image_resized,axis=0)})

        boxes,confs,probs = run_items[0]
        scores = confs *probs
        boxes, scores,labels = utils.nms(boxes,scores,num_classes=1,max_boxes=3,score_thresh=0.5,iou_thresh=0.2)
        img = utils.draw_boxes(image,boxes,scores,labels,CLASSES,[IMAGE_H,IMAGE_W],show=True)

test()
#problem: predicted boxes have negative infinite numbers.