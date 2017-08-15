#!/usr/bin/env python2
# coding: utf-8

import os
import cv2
import argparse
import math
import numpy as np

def IOU(bbox_pre, bbox_gt):
  #print bbox_pre
  #print bbox_gt
  maxX1 = max(bbox_pre[0], bbox_gt[0])
  maxY1 = max(bbox_pre[1], bbox_gt[1])
  minX2 = min(bbox_pre[2], bbox_gt[2])
  minY2 = min(bbox_pre[3], bbox_gt[3])
  w_inter = max(minX2 - maxX1 + 1, 0)
  h_inter = max(minY2 - maxY1 + 1, 0)
  inter = w_inter * h_inter
  w_gt = max(bbox_gt[2] - bbox_gt[0] + 1, 0)
  h_gt = max(bbox_gt[3] - bbox_gt[1] + 1, 0)
  w_pre = max(bbox_pre[2] - bbox_pre[0] + 1, 0)
  h_pre = max(bbox_pre[3] - bbox_pre[1] + 1, 0)
  return inter / (w_gt * h_gt + w_pre * h_pre - inter)

def align2bbox(pts):
  # [x0 y0 x1 y1 x2 y2 x3 y3 x4 y4]
  bbox = []
  x_center = (pts[0] + pts[2] + pts[6] + pts[8]) / 4
  y_center = (pts[1] + pts[3] + pts[7] + pts[9]) / 4
  w = pts[2] - pts[0] + pts[8] - pts[6]
  h = pts[7] - pts[1] + pts[9] - pts[3]
  bbox.append(x_center - w / 2)
  bbox.append(y_center - h / 2)
  bbox.append(x_center + w / 2)
  bbox.append(y_center + h / 2)
  return bbox

def pre_parse(pre_list):
  # parse bbox and landmark
  dic = {}
  bbox_count = 0
  count3 =  0
  count_name =0
  f = open(pre_list, 'r')
  name = f.next()
  while name:
    count_name = count_name + 1
    name = name.strip().split('_')[0]
    count = int(f.next().strip())
    bbox_count = bbox_count + count
    pairs = []
    for i in range(count):
      pair = {}
      line1 = f.next()
      line2 = f.next()
      bbox = [float(x) for x in line1.strip().split(' ')]
      pts = [float(x) for x in line2.strip().split(' ')]
      assert len(bbox) == 4
      assert len(pts) == 10
      pair['bbox'] = bbox[0:4]
      pair['landmark'] = pts[0:10]
      pairs.append(pair)
    dic[name] = pairs
    count3 = count3 + len(pairs)
    try:
      name = f.next()
    except StopIteration:
      break
  print "b_count = ", bbox_count
  count = 0
  count_name1 = 0
  for k in dic:
    count_name1 = count_name1 + 1
    count = count + len(dic[k])
  print "count2 = ", count
  print "count3 = ", count3
  print count_name
  print count_name1
  print len(dic)
  return (dic, bbox_count)

def gt_parse(bbox_gt_list, pts_file_list, pts_file_path):
  # parse bbox
  dic = {}
  bbox_count = 0
  f = open(bbox_gt_list, 'r')
  name = f.next()
  while name:
    name = name.strip().split('_')[0]
    count = int(f.next().strip())
    pairs = []
    for i in range(count):
      pair = {}
      line1 = f.next()
      bbox = [float(x) for x in line1.strip().split(' ')]
      assert len(bbox) == 5
      pair['bbox'] = bbox[0:4]
      pairs.append(pair)
      bbox_count = bbox_count + 1
    dic[name] = pairs
    try:
      name = f.next()
    except StopIteration:
      break
  f.close()

  # parse landmark
  with open(pts_file_list, 'r') as f:
    pts_files = f.readlines()
  for file in pts_files:
    name = file.strip().split('_')[0]
    file_name = file.strip() + '.pts'
    with open(os.path.join(pts_file_path, file_name), 'r') as fp:
      info = fp.readlines()
    #print info[1].strip().split(' ')
    #print info[1].strip().split(' ')[1]
    num_pts = int(info[1].strip().split('  ')[1])
    assert num_pts == 5
    pts = []
    for i in range(num_pts):
      pts.append(float(info[3 + i].strip().split(' ')[0]))
      pts.append(float(info[3 + i].strip().split(' ')[1]))
    # check belongings
    ious = []
    if dic.has_key(name):
      for i in range(len(dic[name])):
        bbox = dic[name][i]['bbox']
        bbox_from_pts = align2bbox(pts)
        ious.append(IOU(bbox, bbox_from_pts))
      print "max(ious)", ious
      print name
      if ious:
        if  max(ious) > 0.3:
          dic[name][ious.index(max(ious))]['landmark'] = pts
  return (dic, bbox_count)

def main():
  parser = argparse.ArgumentParser(description = 'Compute bbox and landmark accuracy')
  parser.add_argument('bbox_gt_list', help = 'Ground truth list file')
  parser.add_argument('pts_file_list', help = 'Landmark file list')
  parser.add_argument('pts_file_path', help = 'Landmark file path')
  parser.add_argument('pre_list', help = 'predict list file')
  args = parser.parse_args()

  bbox_gt_list = args.bbox_gt_list
  pts_file_list = args.pts_file_list
  pts_file_path = args.pts_file_path
  pre_list = args.pre_list

  gt_dic, gt_bbox_count = gt_parse(bbox_gt_list, pts_file_list, pts_file_path)
  pre_dic, pre_bbox_count  = pre_parse(pre_list)

  # caculate accuracy
  pre_all = 0
  pre_true = 0
  pre_false = 0
  gt_true = 0
  landmark_count = 0
  losses = []
  for k in pre_dic.keys():
    #gt_list = gt_dic[k]
    #pre_list = pre_dic[k]
    # pre_all = pre_all + len(pre_dic[k])
    for pair in pre_dic[k]:
      pre_all = pre_all + 1
      ious = []
      for pair_gt in gt_dic[k]:
        ious.append(IOU(pair['bbox'], pair_gt['bbox']))
      if ious:
        if max(ious) > 0.5:
          pre_true = pre_true + 1
          index = ious.index(max(ious))
          if gt_dic[k][index].has_key('landmark'):
            pts_gt = gt_dic[k][index]['landmark']
            pts_pre = pair['landmark']
            landmark_count = landmark_count + 1
            loss = 0
            a_eye = (pts_gt[0] - pts_gt[2]) * (pts_gt[0] - pts_gt[2])
            b_eye = (pts_gt[1] - pts_gt[3]) * (pts_gt[1] - pts_gt[3])
            l_eye = math.sqrt(a_eye + b_eye)
            for c in range(5):
              a = (pts_gt[2 * c] - pts_pre[2 * c]) * (pts_gt[2 * c] - pts_pre[2 * c])
              b = (pts_gt[2 * c + 1] - pts_pre[2 * c + 1]) * (pts_gt[2 * c + 1] - pts_pre[2 * c + 1])
              loss = loss + math.sqrt(a + b) / l_eye
            losses.append(loss / 5)

  #assert pre_all == pre_bbox_count
  print "pre_all = ", pre_all
  print "pre_bbox_count = ", pre_bbox_count
  print "gt_bbox_count = ", gt_bbox_count
  print "pre_true = ", pre_true
  print "accuracy = ", float(pre_true) / float(pre_bbox_count)
  print "recall = ", float(pre_true) / float(gt_bbox_count)
  print "loss_len = ", len(losses)
  #print losses
  print "mean loss = ", sum(losses) / len(losses)


if __name__ == '__main__':
  main()
