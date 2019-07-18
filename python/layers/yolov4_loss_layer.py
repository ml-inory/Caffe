# coding: utf-8
'''
/******************************************************************************

        版权所有 (C),2018-2025  开放智能机器（上海）有限公司
                        OPEN AI LAB
                        
******************************************************************************
  文 件 名   : yolov4_loss_layer.py
  作     者:  杨荣钊
  生成日期   :  2019年6月15日
  功能描述   :  YOLOv4数据层
******************************************************************************/ 
'''
import os
import sys
import cv2
import time
sys.path.append('/home/rzyang/caffe/build/python')
import caffe
import numpy as np


# ===========================================================
#            YOLOV4_LOSS_LAYER
# 功能描述: 损失层
# 输入参数: 
# ===========================================================
class YOLOV4_LOSS_LAYER(caffe.Layer):
    # =====================================
    # 函数名称: setup
    # 功能描述: 检查输入的参数是否合法
    # 输入参数: 
    #           bottom    上一层
    #           top       下一层 data label
    # 输出参数:
    # 返回参数: 
    # =====================================
    def setup(self, bottom, top):
        # top出3个
        if len(top) != 1:
            raise Exception('YOLOV4_LOSS_LAYER Error: top shape must be 1!')
        if len(bottom) != 2:
            raise Exception('YOLOV4_LOSS_LAYER Error: bottom shape must be 2!')

        # 解析参数
        self.batch_size = bottom[0].num
        self.face_num = bottom[1].data.shape[2]
        self.diff = np.zeros((self.batch_size, 1, self.face_num, 4), dtype=np.float32)
        self.lamb = 1e-5


    def reshape(self, bottom, top):
        top[0].reshape(1)


    def forward(self, bottom, top):
        # bottom: mbox_loc label  
        # print bottom[1].data.shape
        self.diff = bottom[0].data[...] - bottom[1].data[...]
        top[0].data[...] = np.sum(self.diff**2) / self.batch_size / 2.0
        # top[0].data[...] = np.sum(self.diff**2) / self.batch_size / 2.0 + self.lamb * np.sum(np.abs(bottom[0].data[...]))


    def backward(self, top, propagate_down, bottom):
        # print self.diff.shape
        # print bottom[0].diff.shape
        bottom[0].diff[...] = self.diff / self.batch_size
        # bottom[0].diff[...] = self.diff / self.batch_size + self.lamb * np.abs(bottom[0].data[...])
        # print bottom[0].data[0, :4, ...].flatten(), bottom[1].data[0, :4, ...].flatten()