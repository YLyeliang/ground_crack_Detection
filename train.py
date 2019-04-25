
import os
import tensorflow as tf
from utility import utils,yolov3
from data.dataset import dataset,Parser
sess=tf.Session()

IMAGE_H, IMAGE_W = 416,416
BATCH_SIZE =8
STEPS = 30000
LR = 0.0001
DECAY_STEP = 1000
DECAY_RATE =0.9
SHUFFLE_SIZE = 100
CLASSES = {0:'crack'}
ANCHORS = utils.get_anchors('./data/anchors.txt',IMAGE_H,IMAGE_W)
NUM_CLASSES = len(CLASSES)
EVAL_INTERNAL = 100
SAVE_INTERNAL =1000
SAVE_PATH="/home/yel/PycharmProjects/lab/"

train_rfrecord = "/home/yel/data/Aerialgoaf/train.tfrecords"
test_tfrecord = "/home/yel/data/Aerialgoaf/test.tfrecords"

parser = Parser(IMAGE_H, IMAGE_W, ANCHORS, NUM_CLASSES)
trainset = dataset(parser, train_rfrecord, BATCH_SIZE, shuffle=SHUFFLE_SIZE)
testset = dataset(parser, test_tfrecord, BATCH_SIZE, shuffle=None)

is_training = tf.placeholder(tf.bool)
example = tf.cond(is_training, lambda: trainset.get_next(), lambda: testset.get_next())

def training():
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

    global_step = tf.Variable(0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    write_op = tf.summary.merge_all()
    writer_train = tf.summary.FileWriter("./data/train")

    # update_vars = tf.contrib.framework.get_variables_to_restore(include=["yolov3/yolo-v3"])
    learning_rate = tf.train.exponential_decay(LR, global_step, DECAY_STEP, DECAY_RATE, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # set dependencies for BN ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss[0],global_step=global_step)

    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    # saver_to_restore.restore(sess, "./checkpoint/yolov3.ckpt")
    saver = tf.train.Saver(max_to_keep=3)

    for step in range(STEPS):
        run_items = sess.run([train_op, write_op, y_pred, y_true] + loss, feed_dict={is_training: True})

        if (step + 1) % EVAL_INTERNAL == 0:
            train_rec_value, train_prec_value = utils.evaluate(run_items[2],run_items[3])

        writer_train.add_summary(run_items[1], global_step=step)
        writer_train.flush()  # flushes the event file to disk
        if (step + 1) % SAVE_INTERNAL == 0: saver.save(sess, save_path=os.path.join(SAVE_PATH,'model.ckpt'), global_step=step + 1)

        print("=> STEP %10d [TRAIN]:\tloss_xy:%7.4f \tloss_wh:%7.4f \tloss_conf:%7.4f \tloss_class:%7.4f"
              % (step + 1, run_items[5], run_items[6], run_items[7], run_items[8]))

training()