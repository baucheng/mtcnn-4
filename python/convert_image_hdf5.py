#!/usr/bin/env python
# coding=utf-8

import os
import cv2
import h5py
import argparse
import numpy as np
import random

def write_hdf5(file, data, label_class, label_bbox, label_landmarks):
  # transform to np array
  data_arr = np.array(data, dtype = np.float32)
  # print data_arr.shape
  # if no swapaxes, transpose to num * channel * width * height ???
  # data_arr = data_arr.transpose(0, 3, 2, 1)
  label_class_arr = np.array(label_class, dtype = np.float32)
  label_bbox_arr = np.array(label_bbox, dtype = np.float32)
  label_landmarks_arr = np.array(label_landmarks, dtype = np.float32)
  with h5py.File(file, 'w') as f:
    f['data'] = data_arr
    f['label_class'] = label_class_arr
    f['label_bbox'] = label_bbox_arr
    f['label_landmarks'] = label_landmarks_arr

# list_file format:
# image_path | label_class | label_boundingbox(4) | label_landmarks(10)
def convert_dataset_to_hdf5(list_file, path_data, path_save,
                            size_hdf5, tag):
  with open(list_file, 'r') as f:
    annotations = f.readlines()
  num = len(annotations)
  print "%d pics in total" % num
  random.shuffle(annotations)

  data = []
  label_class = []
  label_bbox = []
  label_landmarks = []
  count_data = 0
  count_hdf5 = 0
  for line in annotations:
    line_split = line.strip().split(' ')
    assert len(line_split) == 16
    path_full = os.path.join(path_data, line_split[0])
    datum = cv2.imread(path_full)
    classes = float(line_split[1])
    bbox = [float(x) for x in line_split[2:6]]
    landmarks = [float(x) for x in line_split[6:]]
    # print line_split[0]
    # print "classes", classes
    # print "bbox", bbox
    # print "landmarks", landmarks
    if datum is None:
      continue
    # normalization
    # BGR to RGB
    tmp = datum[:, :, 2].copy()
    datum[:, :, 2] = datum[:, :, 0]
    datum[:, :, 0] = tmp
    datum = (datum - 127.5) * 0.0078125 # [0,255] -> [-1,1]
    datum = np.swapaxes(datum, 0, 2)
    data.append(datum)
    label_class.append(classes)
    label_bbox.append(bbox)
    label_landmarks.append(landmarks)
    count_data = count_data + 1
    if 0 == count_data % size_hdf5:
      path_hdf5 = os.path.join(path_save, tag + str(count_hdf5) + ".h5")
      write_hdf5(path_hdf5, data, label_class, label_bbox, label_landmarks)
      count_hdf5 = count_hdf5 + 1
      data = []
      label_class = []
      label_bbox = []
      label_landmarks = []
      print count_data
  # handle the rest
  if data:
    path_hdf5 = os.path.join(path_save, tag + str(count_hdf5) + ".h5")
    write_hdf5(path_hdf5, data, label_class, label_bbox, label_landmarks)
  print "count_data: %d" % count_data

def main():
  parser = argparse.ArgumentParser(description = 'Convert dataset to hdf5')
  parser.add_argument('list_file', help = 'Special format list file')
  parser.add_argument('path_data', help = 'Path to original dataset')
  parser.add_argument('path_save', help = 'Path to save hdf5')
  parser.add_argument('-s', '--size_hdf5', type = int,
                      help = 'Batch size of hdf5, Default: 2048')
  parser.add_argument('-t', '--tag', type = str,
                      help = 'Specify train, test or validation, Default: train_')
  parser.set_defaults(size_hdf5 = 256, tag = 'train_')
  args = parser.parse_args()

  list_file = args.list_file
  path_data = args.path_data
  path_save = args.path_save
  size_hdf5 = args.size_hdf5
  tag = args.tag

  assert os.path.exists(path_data)
  if not os.path.exists(path_save):
    os.makedirs(path_save)
  assert size_hdf5 > 0

  # convert
  convert_dataset_to_hdf5(list_file, path_data, path_save, size_hdf5, tag)


if __name__ == '__main__':
  main()
