#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:06:50 2020

@author: ahmedbingol
"""

import numpy as np
import cv2 as cv
import time

folder="/Users/ahmedbingol/Desktop/Hw2/"
img = cv.imread('/Users/ahmedbingol/Desktop/SunnyLake.bmp')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans() for 4 different K
start_time = time.time()

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K1 = 4
ret,label,center=cv.kmeans(Z,K1,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
labels_unique = np.unique(label)
print("Cluster no for " , K1 ,"is = " ,len(labels_unique))

K2 = 8
ret2,label2,center2=cv.kmeans(Z,K2,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
labels_unique = np.unique(label2)
print("Cluster no for " , K2 ,"is = " ,len(labels_unique))


K3 = 16
ret3,label3,center3=cv.kmeans(Z,K3,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

labels_unique = np.unique(label3)
print("Cluster no for " , K3 ,"is = " ,len(labels_unique))

K4 = 32
ret4,label4,center4=cv.kmeans(Z,K4,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
labels_unique = np.unique(label4)
print("Cluster no for " , K4 ,"is = " ,len(labels_unique))

K5 = 64
ret5,label5,center5=cv.kmeans(Z,K5,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
labels_unique = np.unique(label5)
print("Cluster no for " , K5 ,"is = " ,len(labels_unique))


# Now convert back into uint8, and make original image
center = np.uint8(center)
mask = center[label.flatten()]
res1 = mask.reshape((img.shape))

center2= np.uint8(center2)
mask = center2[label2.flatten()]
res2 = mask.reshape((img.shape))

center3= np.uint8(center3)
mask = center3[label3.flatten()]
res3 = mask.reshape((img.shape))

center4= np.uint8(center4)
mask = center4[label4.flatten()]
res4 = mask.reshape((img.shape))

center5= np.uint8(center5)
mask = center5[label5.flatten()]
res5 = mask.reshape((img.shape))

cv.imwrite(folder +"K_mean_output_K_is"+str(K1)+ ".bmp",res1)
cv.imwrite(folder +"K_mean_output_K_is"+str(K2)+ ".bmp",res2)
cv.imwrite(folder +"K_mean_output_K_is"+str(K3)+ ".bmp",res3)
cv.imwrite(folder +"K_mean_output_K_is"+str(K4)+ ".bmp",res4)
cv.imwrite(folder +"K_mean_output_K_is"+str(K5)+ ".bmp",res5)

print("--- %s seconds required for " %(time.time() - start_time))

cv.imshow('Original',img)
cv.imshow('res1',res1)
cv.imshow('res2',res2)
cv.imshow('res3',res3)
cv.imshow('res4',res4)
cv.waitKey(0)
cv.destroyAllWindows()