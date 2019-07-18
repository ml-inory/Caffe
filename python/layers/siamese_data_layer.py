# coding: utf-8
'''
/******************************************************************************

        版权所有 (C),2018-2025  开放智能机器（上海）有限公司
                        OPEN AI LAB
                        
******************************************************************************
  文 件 名   : siamese_data_layer.py
  作     者:  杨荣钊
  生成日期   :  2019年6月12日
  功能描述   :  SiameseFC的数据输入层
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
#            SiameseDataLayer
# 功能描述: 通过指定input、templar、output的txt地址，输出N*3张图片
# 输入参数: 
#           batch_size    一次输出多少个数据
# ===========================================================
class SiameseDataLayer(caffe.Layer):
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
        # top出3个
        if len(top) != 2:
            raise Exception('SiameseDataLayer Error: top shape must be 3!')

        # 定义输出名称
        self.top_names = ['data', 'label']
        # 解析参数
        self.params = eval(self.param_str)
        param_keys = self.params.keys()
        if 'batch_size' not in param_keys:
            raise Exception('SiameseDataLayer Error: batch_size must be given')
        else:
            self.batch_size = int(self.params['batch_size'])

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        print 'SiameseDataLayer forward'
        top[0].data[...] = np.zeros((self.batch_size, 1, 1, 1))
        top[1].data[...] = np.zeros((self.batch_size, 1, 1, 1))
        # top[2].data[...] = np.zeros((self.batch_size, 1, 1, 1))

    def backward(self, bottom, top):
        print 'SiameseDataLayer backward'