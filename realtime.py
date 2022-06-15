#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/13 15:20
# @Author  : shen.p
# @Site    : 
# @File    : realtime.py
# @Software: PyCharm
import cv2
import time
import numpy as np

from bodyclassfy import PoseClassifier
from count import RepetitionCounter
from data_process import show_image
from posture import EMADictSmoothing, FullBodyPoseEmbedder
from visual import PoseClassificationVisualizer

if __name__ == '__main__':

    # 获取摄像头，传入0表示获取系统默认摄像头
    cap = cv2.VideoCapture(0)

    # 打开cap
    # cap.open(0)

    # 无限循环，直到break被触发
    while cap.isOpened():
        # 获取画面
        success, frame = cap.read()
        if not success:
            break
        ## !!!处理帧函数
        # frame = process_frame(frame)

        # 展示处理后的三通道图像
        cv2.imshow('my_window', frame)

        if cv2.waitKey(1) in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
            break

    # 关闭摄像头
    cap.release()

    # 关闭图像窗口
    cv2.destroyAllWindows()