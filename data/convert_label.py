# reading voc format xml files,and write the bboxes into a txt file.
# the format of txt file is path,xmin,ymin,xmax,ymax,class_id ...
#

import xml.etree.ElementTree as ET
import argparse
import os
import random
import cv2
classes= ["crack"]

parser =argparse.ArgumentParser()
parser.add_argument("--train",default="/home/yel/data/Aerialgoaf/detection/train")
parser.add_argument("--test",default="/home/yel/data/Aerialgoaf/detection/test")
parser.add_argument("--txt_path",default="/home/yel/data/Aerialgoaf/detection")
flags=parser.parse_args()

sets=['train','test']

def train_test_select(path):
    list=os.listdir(path)
    list.remove('train')
    list.remove('test')
    files=random.sample(list,39)
    for i in files:
        src_img=os.path.join(path,i)

# def jpg_to_png(path):

def conver_annotation(path,txt_file):
    img_list = os.listdir(path)
    for i in img_list:
        if 'xml' not in i:
            continue
        img_path="haze"+'/'+ i.split(".")[0]+".png"
        txt_file.write(os.path.join(path,img_path))     #write image_path
        tree = ET.parse(os.path.join(path, i))
        root = tree.getroot()
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult)==1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            txt_file.write(" "+ " ".join([str(a) for a in b])+ " "+str(cls_id))
        txt_file.write('\n')

for set in sets:
    txt_file_path = os.path.join(flags.txt_path,"{}.txt".format(set))
    txt_file = open(txt_file_path,'w')
    if 'train' in set:
        conver_annotation(flags.train,txt_file)
        txt_file.close()
    else:
        conver_annotation(flags.test,txt_file)
        txt_file.close()



