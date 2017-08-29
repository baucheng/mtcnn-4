import cv2
import argparse
import os
import random

def show_bbox_landmark(list_file, path_data):
  with open(list_file, 'r') as f:
    annotations = f.readlines()
  num = len(annotations)
  print "%d pics in total" % num
  # random.shuffle(annotations)

  for line in annotations:
    line_split = line.strip().split(' ')
    print line_split[0]
    path_full = os.path.join(path_data, line_split[0])
    datum = cv2.imread(path_full)
    classes = float(line_split[1])
    bbox = [float(x) for x in line_split[2:6]]
    landmarks = [float(x) for x in line_split[6:]]
    print classes
    print bbox
    print landmarks

    (h, w, c) = datum.shape

    if (bbox[0] != -1):
      x1 = bbox[0] * w
      y1 = bbox[1] * h
      x2 = bbox[2] * w + w
      y2 = bbox[3] * h + h
      cv2.rectangle(datum, (int(x1), int(y1)), (int(x2), int(y2)),
                    (0, 255, 0), 1)

    if (landmarks[0] != -1):
      for i in range(5):
        cv2.circle(datum, (int(landmarks[i] * w), int(landmarks[i + 5] * h)),
                   2, (255, 0, 0))
    cv2.imshow(str(line_split[0]), datum)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description =
      "Display bbox and landmarks to check label")
  parser.add_argument('list_file', help = 'Special format list file')
  parser.add_argument('path_data', help = 'Path to original dataset')

  args = parser.parse_args()
  list_file = args.list_file
  path_data = args.path_data

  assert os.path.exists(path_data)
  show_bbox_landmark(list_file, path_data)
