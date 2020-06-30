#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 14:36:40 2020

@author: ahmedbingol
"""

import numpy as np
from sklearn.cluster import MeanShift
import cv2 
import time

filename= '/Users/ahmedbingol/Desktop/SunnyLake.bmp'
original = cv2.imread(filename,cv2.IMREAD_COLOR)

folder="/Users/ahmedbingol/Desktop/Hw2/"
minc= 8
maxc= 200
resize_crit =False
scalefactor=1/5

#One option to speed up process is making smaller to picture
if (resize_crit == True):
    height, width = original.shape[:2]
    original = cv2.resize(original,(int(scalefactor*width), int(scalefactor*height)), interpolation = cv2.INTER_NEAREST)
        
# convierte la imagen en un arreglo
original = np.array(original)
        
# Reshaping to Nx1 
X = np.reshape(original, [-1, 3])
        
# Bandwith
bandwidth=minc

start_time = time.time()

while bandwidth <= maxc:

    print("Bandwidth ",bandwidth)
            
    # Performing mean shift with given bandwidth and fitting to data
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=1)
    ms.fit(X)
   
    labels = ms.labels_   
    cluster_centers = ms.cluster_centers_
    
    # converts each cluster center to a 3 channel 8 bit image
    cluster_centers= cluster_centers.astype(np.uint8)
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    # Print the number of clusters
    print("Clusters number is : %d" % n_clusters_)        

    # Combine each cluster with the label of each pixel
    result= cluster_centers [labels.flatten()]
    
    # Resize to original image 
    img_meanshift=result.reshape((original.shape))  
    cv2.imshow( str(bandwidth)+ "Output",img_meanshift)
    #cv2.waitKey(0)
    
    # Writing result of each epoch to given folder directory 
    cv2.imwrite(folder+"bandwidth"+str(bandwidth)+ "clustersize"+ str(n_clusters_) +" .bmp",img_meanshift)
    
    # double the bandwidth for a next iteration	
    bandwidth=bandwidth*2

print("--- %s seconds required for " %(time.time() - start_time)," Start bandwith ", minc, "End bandwidth", bandwidth)    
cv2.destroyAllWindows()
