#!/usr/bin/env python
# coding=utf-8

import cv2
import h5py
import numpy as np

def wirte_hdf5(file, data, label_class, label_bbox, label_landmarks):
  with h5py.File(file, 'w') as f:
    f['data'] = data
    f['class'] = label_class
    f['bbox'] = label_bbox
    f['landmarks'] = label_landmarks

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
      if not datum:
        continue
      data.append(datum)
      label_class.append(classes)
      label_bbox.append(bbox)
      label_landmarks.append(landmarks)
      # transform to np array
      data = np.array(data, dtype = np.float32)
      # num * channel * height * width
      data = data.transpose(0, 3, 1, 2)
      label_class = np.array(label_class, dtype = np.float32)
      label_bbox = np.array(label_bbox, dtype = np.float32)
      label_landmarks = np.array(label_landmarks, dtype = np.float32)
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
    path_hdf5 = path_save + tag + str(count_hdf5) + ".h5"
    wirte_hdf5(path_hdf5, data, label_class, label_bbox, label_landmarks)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(discription = 'Convert dataset to hdf5')
  parser.add_argument('list_file', help = 'Special format list file')
  parser.add_argument('path_data', help = 'Path to dataset')
  parser.add_argument('path_save', help = 'Path to save')
  parser.add_argument('-s', '--size_hdf5', type = int,
                      help = 'Size of hdf5, Default: 2048')
  parser.add_argument('-t', '--tag', type = str,
                      help = 'Specify train test or validation, Default: train_')
  parser.set_defaults(size_hdf5 = 2048, tag = 'train_')
  args = parser.pase_args()

  list_file = args.list_file
  path_data = args.path_data
  path_save = args.path_save
  size_hdf5 = args.size_hdf5
  tag = args.tag

  assert os.path.exists(path_dataset)
  if not os.path.exists(path_save):
    os.makedirs(path_save)
  assert size_hdf5 > 0

  # convert
  convert_dataset_to_hdf5(list_file, path_data, path_save, size_hdf5, tag)
