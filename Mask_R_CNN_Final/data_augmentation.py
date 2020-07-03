#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:27:41 2020

@author: ahmedbingol
"""

# Importing necessary library 
import Augmentor 
# Passing the path of the image directory 
p = Augmentor.Pipeline("/Users/ahmedbingol/Desktop/bbl588_final/data") 
  

#   AUGMENTATION PARAMETERS


#  %50 proability to flip  
p.flip_left_right(0.5) 
p.flip_top_bottom(0.2)


# %15 probability of change of brigthness between 0.20 to 1.80
p.random_brightness(0.15, 0.20, 1.80)
#  %30 probability to rotate 15 degree left or 15 degree right
p.rotate(0.3, 25, 25)
#  %40 probability to rotate in 90-180-270
p.rotate_random_90(0.15)
# %40 Probability to skew image in different direction 
p.skew(0.4, 0.5) 
# %20 probability to zoom image between 0.8 and 2.5 factor
p.zoom(probability = 0.2, min_factor = 0.8, max_factor = 3.0) 

# Probability of image to be an greyscale image
p.greyscale(0.1)

p.sample(125) 