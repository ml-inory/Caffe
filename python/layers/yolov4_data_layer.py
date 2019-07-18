# coding: utf-8
'''
/******************************************************************************

        版权所有 (C),2018-2025  开放智能机器（上海）有限公司
                        OPEN AI LAB
                        
******************************************************************************
  文 件 名   : yolov4_data_layer.py
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
#            YOLOV4_DATA_LAYER
# 功能描述: 数据层
# 输入参数: 
# ===========================================================
class YOLOV4_DATA_LAYER(caffe.Layer):
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
        if len(top) != 2:
            raise Exception('YOLOV4_DATA_LAYER Error: top shape must be 2!')

        # 解析参数
        self.params = eval(self.param_str)
        print self.params
        self.batch_size = int(self.params['batch_size'])
        self.new_width = int(self.params['new_width'])
        self.new_height = int(self.params['new_height'])
        self.max_face_num = int(self.params['max_face_num'])

        self.batch_loader = BatchLoader(self.params)

        top[0].reshape(self.batch_size, 3, self.new_height, self.new_width)
        top[1].reshape(self.batch_size, self.max_face_num*4, 1, 1)


    def reshape(self, bottom, top):
        pass


    def forward(self, bottom, top):
        for i in xrange(self.batch_size):
            top[0].data[i, ...], top[1].data[i, ...] = self.batch_loader.load_next_image()


    def backward(self, top, propagate_down, bottom):
        pass




class BatchLoader(object):
    def __init__(self, params):
        self.source = params['source']
        self.root_folder = params['root_folder']
        self.new_width = int(params['new_width'])
        self.new_height = int(params['new_height'])
        self.max_face_num = int(params['max_face_num'])

        # 读取所有行
        self.image_list = open(self.source, 'r').read().splitlines()
        self.cur_line = 0


    def load_next_image(self):
        # 是否读取完成
        if self.cur_line == len(self.image_list):
            self.cur_line = 0

        line = self.image_list[self.cur_line]

        # 图像路径
        img_path = os.path.join(self.root_folder, line.split(' ')[0])
        # 读取图片
        img = cv2.imread(os.path.join(self.root_folder, img_path))
        # 原始图片尺寸
        ori_img_h, ori_img_w = img.shape[:2]
        # 缩放
        img = cv2.resize(img, (self.new_width, self.new_height))
        # 转YUV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # HWC->CHW
        img = img.transpose((2, 0, 1))
        
        # 解析框框label x y w h
        bboxes = np.array([int(x) for x in line.split(' ')[1:]]).reshape(-1, 4)
        sorted(bboxes, key=lambda box : box[2]*box[3], reverse=False)
        bboxes[:, 2:] += bboxes[:, 0:2]
        bboxes = bboxes / np.array([ori_img_w, ori_img_h, ori_img_w, ori_img_h], dtype=np.float32)
        bboxes_num = bboxes.shape[0]

        label = np.zeros((self.max_face_num*4, 1, 1), dtype=np.float32)
        if bboxes_num >= self.max_face_num:
            label[:, ...] = bboxes[:self.max_face_num, :].reshape(-1, 1, 1)
        else:
            label[:bboxes_num*4, ...] = bboxes.reshape(-1, 1, 1)

        # 换下一行
        self.cur_line += 1

        return img, label