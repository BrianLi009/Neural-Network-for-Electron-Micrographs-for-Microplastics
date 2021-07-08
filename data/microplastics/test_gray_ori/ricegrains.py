# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:18:56 2019

@author: benjamin
"""

# -*- coding:utf-8 -*-
#本程序用于大津算法的实现
import cv2  #导入opencv模块
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os


print("Hellow word!")     #打印“Hello word！”，验证模块导入成功
#
#
# i = 0
# for path in pathlib.Path("../test_gray").iterdir():
#     if path.is_file():
#        print(path)
#        (filepath, tempfilename) = os.path.split(path)
#        (filename, extension) = os.path.splitext(tempfilename)
#        print(filename)
#        print(type(filename))
#        img = cv2.imread(os.path.join("./", path))
#        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#        gray_output1 = gray[0:512,0:512]
#        gray_output2 = gray[0:512,512:1024]
#        j = 2*i
#        k = 2*i+1
#        path_output1 = os.path.join("./unet20200807/train/label", str(j)+".png")
#        cv2.imwrite(path_output1, gray_output1)
#        path_output2 = os.path.join("./unet20200807/train/label", str(k)+".png")
#        cv2.imwrite(path_output2, gray_output2)
#        i = i + 1

img2 = cv2.imread("../test/27.png")
#img2 = cv2.imread("../test_gray/27_predict.png")
#cv2.namedWindow("imagshow", 2)  
#size = img.shape
#print size
#img = img2[0:850,0:1280,0:3]
img = img2
#img = img2[400:-1,0:1280,0:3]
#img = img2[0:500,0:300,0:3]

#cv2.namedWindow("imagshow", 2)  
#cv2.resizeWindow("imagshow", 640, 450);
#cv2.imshow('imagshow', img)  

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #转换为灰度图


#gray = cv2.GaussianBlur(gray,(11,11),15)
gray = cv2.GaussianBlur(gray,(11,11),15)

#使用局部阈值的大津算法进行图像二值化
#dst = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,101, -1)
#dst = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,51, -5)
dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

'''
cv2.imwrite("pred_otsu.png", dst)
'''
cv2.namedWindow("imagshow", 3)   #创建一个窗口
cv2.resizeWindow("imagshow", 640, 450);
cv2.imshow('imagshow', img)    #显示原始图片k
cv2.namedWindow("dst", 2)   #创建一个窗口
cv2.resizeWindow("dst", 640, 450);
cv2.imshow("dst", dst)  #显示灰度图
cv2.waitKey()




#
# #全局大津算法，效果较差
# #res ,dst = cv2.threshold(gray,0 ,255, cv2.THRESH_OTSU)
#
# element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3, 3))#形态学去噪
# dst=cv2.morphologyEx(dst,cv2.MORPH_OPEN,element)  #开运算去噪
#
#
# contours, hierarchy = cv2.findContours(dst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]  #轮廓检测函数
# cv2.drawContours(dst,contours,-1,(120,0,0),2)  #绘制轮廓
#
# count=0 #米粒总数
# ares_avrg=0  #米粒平均
# #遍历找到的所有米粒
# for cont in contours:
#
#     ares = cv2.contourArea(cont)#计算包围性状的面积
#
#     if ares<50:   #过滤面积小于10的形状 500
#         continue
#     count+=1    #总体计数加1
#     ares_avrg+=ares
#
#     print("{}-blob:{}".format(count,ares),end="  ") #打印出每个米粒的面积
#
#     rect = cv2.boundingRect(cont) #提取矩形坐标
#
#     print("x:{} y:{}".format(rect[0],rect[1]))#打印坐标
#
#     print("x:{} y:{}".format(rect[2],rect[3]))#打印坐标
#
#     cv2.rectangle(img,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,0,0xff),1)#绘制矩形
#
#     y=10 if rect[1]<10 else rect[1] #防止编号到图片之外
#
#     cv2.putText(img,str(count), (rect[0], y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1) #在米粒左上角写上编号
#
# print("print the area:{}".format(round(ares_avrg/ares,2))) #打印出每个米粒的面积
#
# cv2.namedWindow("imagshow", 3)   #创建一个窗口
# cv2.resizeWindow("imagshow", 640, 450);
# cv2.imshow('imagshow', img)    #显示原始图片k
#
# cv2.namedWindow("dst", 2)   #创建一个窗口
# cv2.resizeWindow("dst", 640, 450);
# cv2.imshow("dst", dst)  #显示灰度图
#
#
# #plt.hist(gray.ravel(), 256, [0, 256]) #计算灰度直方图
# #plt.show()


cv2.waitKey()
