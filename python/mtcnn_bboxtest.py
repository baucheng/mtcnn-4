#!/usr/bin/env python2.7
# coding: utf-8

import cv2
import numpy as np


def computer_IOU(bbox_pre, bbox_gt):
    x1 = bbox_pre[0]
    y1 = bbox_pre[1]
    x2 = bbox_pre[2]
    y2 = bbox_pre[3]

    iou = []
    for i in range(bbox_gt.shape[0]):
        # the iou
        maxX = max(x1, bbox_gt[i][0])
        maxY = max(y1, bbox_gt[i][1])
        minX = min(x2, bbox_gt[i][2])
        minY = min(y2, bbox_gt[i][3])
        # maxx1 and maxy1 resume
        maxX = max(minX - maxX + 1, 0)
        maxY = max(minY - maxY + 1, 0)
        IOU = maxX * maxY
        w_gt = max(bbox_gt[i][2]-bbox_gt[i][0] + 1, 0)
        h_gt = max(bbox_gt[i][3] - bbox_gt[i][1] + 1, 0)
        w_pre = max(x2 - x1 + 1, 0)
        h_pre = max(y2 - y1 + 1, 0)
        IOU = IOU / ( w_gt * h_gt + w_pre * h_pre - IOU )
        iou.append(IOU)

    return iou



img_dir = '/home/dkk/mtcnn/DuinoDu/test/afw_image/'

# 判断这是否为一个主程序，其他python程序无法调用
if __name__ == '__main__':

    pre_all = 0
    pre_true = 0
    pre_false = 0
    gt = open('./bbox_gt.txt', 'r')
    pre = open('./bbox_pre_DuinoDu.txt', 'r')
    pre_error = open('./error_list.txt', 'w')
    name_gt = gt.next()
    name_pre = pre.next()

    while name_gt:
        # ground truth bbox
        name_gt = name_gt[:-1]
        img = cv2.imread(img_dir+name_gt)

        bboxes_gt = []
        num_gt = gt.next()
        num_gt = int(num_gt[:-1])
        if num_gt == 0:
            name_gt = gt.next()
            num_pre = pre.next()
            num_pre = int(num_pre[:-1])
            for i in range(num_pre):
                comp = pre.next()
            name_pre = pre.next()
            continue

        for i in range(num_gt):
            comp = gt.next()
            bbox = comp.strip().split(' ')
            bboxes_gt.append([float(bbox[0]),float(bbox[1]),float(bbox[2]),float(bbox[3])])

        bboxes_gt = np.array(bboxes_gt)

        # pre bbox
        name_pre = name_pre[:-1]
        bboxes_pre = []
        num_pre = pre.next()
        num_pre = int(num_pre[:-1])
        pre_all = pre_all + num_pre
        if num_pre == 0:
            name_gt =gt.next()
            name_pre = pre.next()
            continue

        for i in range(num_pre):
            comp = pre.next()
            bbox = comp.strip().split(' ')
            bboxes_pre.append([float(bbox[0]),float(bbox[1]),float(bbox[2]),float(bbox[3])])

        bboxes_pre = np.array(bboxes_pre)

        # computer iou
        write_error = 0
        for i in range(num_pre):
            IOU = computer_IOU(bboxes_pre[i], bboxes_gt)
            if max(IOU) > 0.5:
                cv2.rectangle(img, (int(bboxes_pre[i][0]), int(bboxes_pre[i][1])), (int(bboxes_pre[i][2]), int(bboxes_pre[i][3])), (255, 0, 0), 3)
                pre_true = pre_true + 1
            else:
                cv2.rectangle(img, (int(bboxes_pre[i][0]), int(bboxes_pre[i][1])), (int(bboxes_pre[i][2]), int(bboxes_pre[i][3])), (0, 255, 0), 3)
                write_error = 1
                pre_false = pre_false + 1
        if write_error == 1:
            pre_error.write(name_pre + '\n')


        for i in range(bboxes_gt.shape[0]):
            cv2.rectangle(img, (int(bboxes_gt[i][0]),int(bboxes_gt[i][1])), (int(bboxes_gt[i][2]),int(bboxes_gt[i][3])),(0, 0, 255), 3)

        save_dir = './results/'
        save_dir = save_dir + name_pre
        cv2.imwrite(save_dir, img)
        cv2.imshow("img", img)
        cv2.waitKey(10)

        try:
            name_gt = gt.next()
            name_pre = pre.next()
            print name_gt
            print name_pre
        except StopIteration:
            print name_gt
            break
    print 'pre_all= ' , pre_all
    print 'pre_true= ', pre_true
    print 'pre_false= ', pre_false
    print 'accuary= ' ,  float(pre_true) / float(pre_all)
    print 'false= ', float(pre_false) / float(pre_all)

    gt.close()
    pre.close()
    pre_error.close()
