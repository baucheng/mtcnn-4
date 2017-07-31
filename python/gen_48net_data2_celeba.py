import sys
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU

stdsize = 48
anno_file = "/home/dkk/projects/caffe_mtcnn/celeba/mtcnn_train_label.txt"
im_dir = "/home/dkk/projects/caffe_mtcnn/celeba/img_celeba"
celeba_save_dir = str(stdsize) + '/celeba'
save_dir = "./" + str(stdsize)

def mkr(dr):
    if not os.path.exists(dr):
        os.mkdir(dr)

mkr(celeba_save_dir)

f4 = open(os.path.join(save_dir, 'celeba_' + str(stdsize) + '.txt'), 'w')
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print "%d pics in total" % num
idx = 0

for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    im_path = os.path.join(im_dir, im_path)
    bbox = map(float, annotation[1:5]) # for wider_face
    pts = map(float, annotation[5:]) # for celeba
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    img = cv2.imread(im_path)
    if idx % 100 == 0:
        print idx, "images done"
    height, width, channel = img.shape
    h_box = bbox[3] - bbox[1]
    w_box = bbox[2] - bbox[0]
    # assert h_box > 0
    # assert w_box > 0
    if (h_box <= 0 or w_box <= 0):
        continue
    diff = h_box - w_box
    if (diff > 0):
        bbox[3] = bbox[3] - diff / 2
        bbox[1] = bbox[1] + diff / 2
    elif (diff < 0):
        bbox[2] = bbox[2] + diff / 2
        bbox[0] = bbox[0] - diff / 2
    pts_bak = pts
    size = bbox[2] - bbox[0]
    assert size == bbox[3] - bbox[1]
    for k in range(len(pts) / 2):
        pts[k*2] = (pts[k*2] - bbox[0]) / float(size);
        pts[k*2+1] = (pts[k*2+1] - bbox[1]) / float(size);

    cropped_im = img[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2]), :]
    resized_im = cv2.resize(cropped_im, (stdsize, stdsize), interpolation=cv2.INTER_LINEAR)
    save_file = os.path.join(celeba_save_dir, "%s.jpg"%idx)
    cv2.imwrite(save_file, resized_im)
    f4.write(str(stdsize)+"/celeba/%s.jpg"%idx + ' -1' + ' -1 -1 -1 -1')
    for k in range(len(pts)):
        f4.write(" %f" % pts[k])
    f4.write("\n")
    idx += 1

f4.close()
