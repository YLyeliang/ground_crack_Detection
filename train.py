
import os
import tensorflow as tf
from utility import utils,yolov3
from data.dataset import dataset,Parser
from PIL import Image
import numpy as np
sess=tf.Session()

IMAGE_H, IMAGE_W = 416,416
BATCH_SIZE =4
STEPS = 30000
LR = 0.0001
DECAY_STEP = 1000
DECAY_RATE =0.9
SHUFFLE_SIZE = 80
CLASSES = {0:'crack'}
ANCHORS = utils.get_anchors('./data/anchors.txt',IMAGE_H,IMAGE_W)
NUM_CLASSES = len(CLASSES)
EVAL_INTERNAL = 100
SAVE_INTERNAL =1000
SAVE_PATH="/home/yel/PycharmProjects/lab/20190505"
FINETUNE_CKPT="/home/yel/PycharmProjects/lab/20190505/model.ckpt-3000"
train_rfrecord = "/home/yel/data/Aerialgoaf/detection/train.tfrecords"
test_tfrecord = "/home/yel/data/Aerialgoaf/detection/test.tfrecords"

parser = Parser(IMAGE_H, IMAGE_W, ANCHORS, NUM_CLASSES)
trainset = dataset(parser, train_rfrecord, BATCH_SIZE, shuffle=SHUFFLE_SIZE)
testset = dataset(parser, test_tfrecord, 1, shuffle=None,repeat=False)

is_training = tf.placeholder(tf.bool)
example = tf.cond(is_training, lambda: trainset.get_next(), lambda: testset.get_next())

def test():
    images, *y_true = example
    model = yolov3.yolov3(NUM_CLASSES, ANCHORS)
    with tf.variable_scope("yolov3"):
        pred_feature_map = model.inference(images, is_training=is_training)
        loss = model.comput_loss(pred_feature_map, y_true)
        y_pred = model.predict(pred_feature_map)

    # sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    saver = tf.train.Saver()
    saver.restore(sess,save_path="/home/yel/PycharmProjects/lab/model.ckpt-21000")

    num = sum(1 for _ in tf.python_io.tf_record_iterator(test_tfrecord))
    for i in range(num):
        run_items = sess.run([ y_pred, y_true] + loss, feed_dict={is_training: False})

        test_rec_value, test_prec_value = utils.evaluate(run_items[0], run_items[1])
        print(
              "=>[TEST]:\trecall:%7.4f \tprecision:%7.4f" % (test_rec_value, test_prec_value))
        # boxes,confs,probs = run_items[0]
        # scores = confs *probs
        # boxes, scores,labels = utils.nms(boxes,scores,num_classes=1)
        # image = utils.draw_boxes(image,boxes,scores,labels,CLASSES,[IMAGE_H,IMAGE_W],show=True)

def training(is_finetune=False):
    images, *y_true = example
    model = yolov3.yolov3(NUM_CLASSES, ANCHORS)
    with tf.variable_scope("yolov3"):
        pred_feature_map = model.inference(images, is_training=is_training)
        loss = model.comput_loss(pred_feature_map, y_true)
        y_pred = model.predict(pred_feature_map)

    tf.summary.scalar("loss/coord_loss", loss[1])
    tf.summary.scalar("loss/sizes_loss", loss[2])
    tf.summary.scalar("loss/confs_loss", loss[3])
    tf.summary.scalar("loss/class_loss", loss[4])

    startstep=0 if not is_finetune else int(FINETUNE_CKPT.split('-')[-1])

    global_step = tf.Variable(0, trainable=False,collections=[tf.GraphKeys.LOCAL_VARIABLES])
    write_op = tf.summary.merge_all()
    writer_train = tf.summary.FileWriter("./data/train/20190505")

    # update_vars = tf.contrib.framework.get_variables_to_restore(include=["yolov3/yolo-v3"])
    learning_rate = tf.train.exponential_decay(LR, global_step, DECAY_STEP, DECAY_RATE, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # set dependencies for BN ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss[0],global_step=global_step)

    saver = tf.train.Saver(max_to_keep=3)
    if is_finetune==True:
        sess.run([tf.local_variables_initializer()])
        saver.restore(sess,FINETUNE_CKPT)
    else:
        sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
    # saver_to_restore.restore(sess, "./checkpoint/yolov3.ckpt")

    for step in range(startstep,startstep+STEPS):
        run_items = sess.run([train_op, write_op, y_pred, y_true] + loss, feed_dict={is_training: True})

        if (step + 1) % EVAL_INTERNAL == 0:
            train_rec_value, train_prec_value = utils.evaluate(run_items[2],run_items[3])
            print(
                "=> STEP %10d [TRAIN]:\trecall:%7.4f \tprecision:%7.4f" % (step + 1, train_rec_value, train_prec_value))

        writer_train.add_summary(run_items[1], global_step=step)
        writer_train.flush()  # flushes the event file to disk
        if (step + 1) % SAVE_INTERNAL == 0: saver.save(sess, save_path=os.path.join(SAVE_PATH,'model.ckpt'), global_step=step + 1)

        if (step+1) % 10 ==0:
            print("=> STEP %10d [TRAIN]:\tloss_xy:%7.4f \tloss_wh:%7.4f \tloss_conf:%7.4f \tloss_class:%7.4f"
              % (step + 1, run_items[5], run_items[6], run_items[7], run_items[8]))

training(is_finetune=True)
# test()