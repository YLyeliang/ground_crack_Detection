
import tensorflow as tf
from utility import utils,yolov3
from data.dataset import dataset,Parser
sess=tf.Session()

IMAGE_H, IMAGE_W = 416,416
BATCH_SIZE =8
STEP = 30000
LR = 0.0001
DECAY_STEP = 1000
DECAY_RATE =0.9
SHUFFLE_SIZE = 100
CLASSES = {0:'crack'}
ANCHORS = utils.
NUM_CLASSES = len(CLASSES)
EVAL_INTERNAL = 100
SAVE_INTERNAL =1000

train_rfrecord = "./"
test_tfrecord = "./"

parser = Parser(IMAGE_H,IMAGE_W,ANCHORS,NUM_CLASSES)
trainset = dataset(parser,train_rfrecord,BATCH_SIZE,shuffle=SHUFFLE_SIZE)
testset = dataset(parser,test_tfrecord,BATCH_SIZE,shuffle=None)

is_training = tf.placeholder(tf.bool)
example = tf.cond(is_training,lambda :trainset.get_next(),lambda :testset.get_next())

images, *y_true = example
model =yolov3.yolov3(NUM_CLASSES,ANCHORS)
