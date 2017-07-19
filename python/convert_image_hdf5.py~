#!/usr/bin/env python
# coding=utf-8

import os
import cv2
import h5py
import argparse
import numpy as np

def write_hdf5(file, data, label_class, label_bbox, label_landmarks):
  # transform to np array
  data_arr = np.array(data, dtype = np.float32)
  print data_arr.shape
  # num * channel * height * width
  data_arr = data_arr.transpose(0, 3, 1, 2)
  label_class_arr = np.array(label_class, dtype = np.float32)
  label_bbox_arr = np.array(label_bbox, dtype = np.float32)
  label_landmarks_arr = np.array(label_landmarks, dtype = np.float32)
  with h5py.File(file, 'w') as f:
    f['data'] = data_arr
    f['label_class'] = label_class_arr
    f['label_bbox'] = label_bbox_arr
    f['label_landmarks'] = label_landmarks_arr

# list_file format:
# image_path | label_class label_boundingbox(4) label_landmarks(10)
def convert_dataset_to_hdf5(list_file, path_data, path_save,
                            size_hdf5, tag):
  with open(list_file, 'r') as file:
    data = []
    label_class = []
    label_bbox = []
    label_landmarks = []
    count_data = 0
    count_hdf5 = 0
    while True:
      line = file.readline()
      if not line:
        break
      line_split = line.split(" ")
      assert 16 == len(line_split)
      path = line_split[0]
      path_full = path_data + path
      datum = cv2.imread(path_full)
      classes = float(line_split[1])
      bbox = [float(x) for x in line_split[2:6]]
      landmarks = [float(x) for x in line_split[6:]]
      if datum is None:
        continue
      data.append(datum)
      label_class.append(classes)
      label_bbox.append(bbox)
      label_landmarks.append(landmarks)
      count_data = count_data + 1
      if 0 == count_data % size_hdf5:
        path_hdf5 = path_save + tag + str(count_hdf5) + ".h5"
        write_hdf5(path_hdf5, data, label_class, label_bbox, label_landmarks)
        count_hdf5 = count_hdf5 + 1
        data = []
        label_class = []
        label_bbox = []
        label_landmarks = []
    # handle the rest
    if data:
      path_hdf5 = path_save + tag + str(count_hdf5) + ".h5"
      write_hdf5(path_hdf5, data, label_class, label_bbox, label_landmarks)


def main():
  parser = argparse.ArgumentParser(description = 'Convert dataset to hdf5')
  parser.add_argument('list_file', help = 'Special format list file')
  parser.add_argument('path_data', help = 'Path to original dataset')
  parser.add_argument('path_save', help = 'Path to save hdf5')
  parser.add_argument('-s', '--size_hdf5', type = int,
                      help = 'Batch size of hdf5, Default: 2048')
  parser.add_argument('-t', '--tag', type = str,
                      help = 'Specify train, test or validation, Default: train_')
  parser.set_defaults(size_hdf5 = 2048, tag = 'train_')
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
