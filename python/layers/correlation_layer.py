# coding: utf-8
'''
/******************************************************************************

        版权所有 (C),2018-2025  开放智能机器（上海）有限公司
                        OPEN AI LAB
                        
******************************************************************************
  文 件 名   : correlation_layer.py
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
#            CorrelationLayer
# 功能描述: 对两个feature map做互相关
# 输入参数: 
# ===========================================================
class CorrelationLayer(caffe.Layer):
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

        W = bottom[0].data
        X = bottom[1].data
        W_c = W.shape[1]
        X_c = X.shape[1]
        self.batch_size = W.shape[0]
        if W_c != X_c:
            raise Exception('CorrelationLayer Error: channel of W(%d) must match X(%d)!' % (W_c, X_c))


    def reshape(self, bottom, top):
        W = bottom[0].data
        X = bottom[1].data
        W_h, W_w = W.shape[2:]
        X_h, X_w = X.shape[2:]

        top[0].reshape(self.batch_size, 1, X_h - W_h + 1, X_w - W_w + 1)


    def forward(self, bottom, top):
        W = bottom[0].data
        X = bottom[1].data
        W_h, W_w = W.shape[2:]
        Z_h, Z_w = top[0].shape[2:]

        for n in xrange(self.batch_size):
            for i in xrange(Z_h):
                for j in xrange(Z_w):
                    top[0].data[n, 0, i, j] = np.sum(W[n] * X[n, :, i:i+W_h, j:j+W_w], dtype=np.float32)
        # print 'W max:', W.max()
        # print 'X max:', X.max()
        # print 'top max:', top[0].data.max()


    def backward(self, top, propagate_down, bottom):
        W = bottom[0].data[...]
        X = bottom[1].data[...]
        W_c, W_h, W_w = W.shape[1:]
        top_h, top_w = top[0].diff.shape[2:]
        bottom[0].diff[...] = 0
        for n in xrange(self.batch_size):
            for i in xrange(top_h):
                for j in xrange(top_w):
                    bottom[0].diff[n, ...] += top[0].diff[n, :, i, j] * X[n, :, i:i+W_h, j:j+W_w]
        bottom[1].diff[...] = 0
        # bottom[0].diff[...] = 1

        # print 'top_diff:', top[0].diff.max()
        # print 'bottom_diff:', bottom[0].diff.max()