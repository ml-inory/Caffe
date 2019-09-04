# coding: utf-8
'''
/******************************************************************************

        版权所有 (C),2018-2025  开放智能机器（上海）有限公司
                        OPEN AI LAB
                        
******************************************************************************
  文 件 名   : retina_loss_layer.py
  作    者   :  杨荣钊
  生成日期   :  2019年7月23日
  功能描述   :  RetinaFace损失层
******************************************************************************/ 
'''
import os
import sys
import cv2
import time
sys.path.append('/home/rzyang/caffe/build/python')
import caffe
import numpy as np

class RetinaLossLayer(caffe.Layer):
    # =====================================
    # 函数名称: setup
    # 功能描述: 检查输入的参数是否合法
    # 输入参数: 
    #           bottom    上一层
    #           top       下一层
    # 输出参数:
    # 返回参数: 
    # =====================================
    def setup(self, bottom, top):
        # bottom要三个, score、bbox、landmark、label
        if len(bottom) != 4:
            raise Exception('RetinaLossLayer Error: bottom shape must be 3!')
        # top出1个
        if len(top) != 1:
            raise Exception('RetinaLossLayer Error: top shape must be 1!')
      
        # 解析参数, stride
        self.params = eval(self.param_str)
        if 'stride' not in self.params.keys():
            raise Exception('RetinaLossLayer Error: must specify param stride')
        if 'pos_iou' not in self.params.keys():
            raise Exception('RetinaLossLayer Error: must specify param pos_iou')
        if 'neg_iou' not in self.params.keys():
            raise Exception('RetinaLossLayer Error: must specify param neg_iou')

        self.batch_size = bottom[0].count
        self.stride = self.params['stride']
        self.pos_iou = self.params['pos_iou']
        self.neg_iou = self.params['neg_iou']

    def reshape(self, bottom, top):
        top[0].reshape(1)
        # 获取内存宽高
        height, width = bottom[0].data.shape[1:3]
        self.softmax_loss = np.zeros((self.batch_size, 1, height, width), dtype=np.float32)

    def forward(self, bottom, top):
        # 获取标签
        label = bottom[3].data
        gt_bbox, gt_landmark = label[..., :4], label[..., 4:]
        # 获取内存宽高
        height, width = bottom[0].data.shape[1:3]
        # 获取概率输出
        prob = bottom[0].data

        # 遍历每个anchor
        for h in xrange(height):
            for w in xrange(width):
                # 生成anchor
                anchor = self.generate_anchor(h, w, self.stride)
                # 计算跟标签框框的最大IOU
                max_iou = calc_max_iou(anchor, gt_bbox)
                # 最大IOU超过pos_iou则认为是正类，小于neg_iou则认为是负类
                if max_iou >= self.pos_iou:
                    self.softmax_loss[:, ] = -prob[:, 1, h, w]
                elif max_iou <= self.neg_iou:
                    softmax_loss = -prob[:, 0, h, w]

    def backward(self, top, propagate_down, bottom):
        pass

    def generate_anchor(self, h, w, stride):
        # x1 y1 x2 y2
        anchor_size = stride / 2
        return (-stride / 2.0 + 0.5 + w*anchor_size, -stride / 2.0 + 0.5 + h*anchor_size, stride / 2.0 + 0.5 + w*anchor_size, stride / 2.0 + 0.5 + h*anchor_size)

    def calc_max_iou(self, anchor, gt_bbox):
        ac_x1, ac_y1, ac_x2, ac_y2 = anchor
        gt_x1 = gt_bbox[:, 0]
        gt_y1 = gt_bbox[:, 1]
        gt_x2 = gt_bbox[:, 2]
        gt_y2 = gt_bbox[:, 3]

        ac_area = (ac_x2 - ac_x1 + 1) * (ac_y2 - ac_y1 + 1)
        gt_areas = (gt_x2 - gt_x1 + 1) * (gt_y2 - gt_y1 + 1)
        x1 = np.maximum(ac_x1, gt_x1)
        y1 = np.maximum(ac_y1, gt_y1)
        x2 = np.minimum(ac_x2, gt_x2)
        y2 = np.minimum(ac_y2, gt_y2)

        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)

        inter = w * h
        iou = inter / (ac_area + gt_areas - inter)

        return iou.max()
