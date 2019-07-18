# coding: utf-8
'''
/******************************************************************************

        版权所有 (C),2018-2025  开放智能机器（上海）有限公司
                        OPEN AI LAB
                        
******************************************************************************
  文 件 名   : sigmoid2d_loss_layer.py
  作     者:  杨荣钊
  生成日期   :  2019年6月12日
  功能描述   :  互相关层
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
#            Sigmoid2DLossLayer
# 功能描述: 2D 交叉熵损失
# 输入参数: 
# ===========================================================
class Sigmoid2DLossLayer(caffe.Layer):
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
        # bottom要两个
        if len(bottom) != 2:
            raise Exception('CorrelationLayer Error: bottom shape must be 3!')
        # top出3个
        if len(top) != 1:
            raise Exception('CorrelationLayer Error: top shape must be 3!')

        self.batch_size = bottom[0].count

    def reshape(self, bottom, top):
        top[0].reshape(1)


    def forward(self, bottom, top):
        label = bottom[1].data
        y = np.zeros(label.shape, dtype=np.float32)
        y[label == 0] = -1
        y[label > 0] = 1

        top[0].data[...] = np.sum(np.log(1 + np.exp(-bottom[0].data[...] * y))) / self.batch_size

        print 'top:', top[0].data[...]


    def backward(self, top, propagate_down, bottom):
        label = bottom[1].data[...]
        y = np.zeros(label.shape, dtype=np.float32)
        y[label == 0] = -1
        y[label > 0] = 1

        mul = -bottom[0].data[...] * y
        exp = np.exp(mul)
        bottom[0].diff[...] = np.sum(1.0 / (1 + exp) * exp * (-y)) / self.batch_size 

        print 'bottom:', bottom[0].diff[...]