# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 11:14:36 2020

@author: felip
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color, io
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle, circle_perimeter
from skimage.util import img_as_ubyte

from skimage.filters.rank import median
from skimage.morphology import disk

from statistics import pstdev

import os


def ball(mean, accums):
    i = 0
    for m, a in zip(mean, accums) :
         if( ( (m > 150.0 and m < 190.0) and a > 0.50 ) or a > 0.8 ):
             return i
         i = i+1
    return -1         

pasta = 'processing/img_recortadas/'

arqs = os.listdir(pasta)

for nome in arqs:
   if nome.endswith('png'): #startswith
        im = io.imread(pasta + nome)
        
         # Utilizar a Guasiana para fazer o filtro com a media
        im2 = im.copy()
        noisy_image = img_as_ubyte(im2[:,:,0])
        noise = np.random.random(noisy_image.shape)
        noisy_image[noise > 0.99] = 255
        noisy_image[noise < 0.01] = 0
        im[:,:,0] = median(noisy_image, disk(1))
        
        noisy_image = img_as_ubyte(im2[:,:,1])
        noise = np.random.random(noisy_image.shape)
        noisy_image[noise > 0.99] = 255
        noisy_image[noise < 0.01] = 0
        im[:,:,1] = median(noisy_image, disk(1))
        
        
        noisy_image = img_as_ubyte(im2[:,:,2])
        noise = np.random.random(noisy_image.shape)
        noisy_image[noise > 0.99] = 255
        noisy_image[noise < 0.01] = 0
        im[:,:,2] = median(noisy_image, disk(1))
        ''' plt.figure()
        plt.imshow(im)
        plt.show()'''
        # Load picture and detect edges
        image = img_as_ubyte(im[:,:,0])
        edges = canny(image, sigma=3, low_threshold=30, high_threshold=85)
        '''plt.imshow(edges, cmap=plt.cm.gray)
        plt.show()   '''
        # Detect two radii
        hough_radii = np.arange(3, 15, 1)
        hough_res = hough_circle(edges, hough_radii)
        
        
        # Select the most prominent 3 circles
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                   total_num_peaks=3)
        mean = np.mean(im[cy[:],cx[:],], axis=1)
        
        # Draw them
        i = 0
        for center_y, center_x, radius, ac in zip(cy, cx, radii, accums):
            #accums, center_x, center_y, radius = accums[0], cx[0], cy[0], radii[0]
            image = color.gray2rgb(image)
            circy, circx = circle(center_y, center_x, radius,
                                            shape=image.shape)
            print(nome +' - accums: ' +str(round(ac,2)) + ' - mean: '+ str(round(im[circy, circx, :].mean(),2)) + ' - dvp: '+ str( round(  np.std(im[circy, circx, :]),2)))
            #image[circy, circx] = 255
            
            #circy, circx = circle_perimeter(center_y, center_x, radius,
            #                           shape=image.shape)
            #print(im[center_y, center_x, :])
            #image[circy, circx] = (255,0,0)
            #plt.text(circx[0], circy[0], str(i), bbox=dict(facecolor='red', alpha=0.5))
            
            #and np.std(im[circy, circx, :]) > 20 and np.std(im[circy, circx, :]) < 45 
            i = i+1
            if(im[circy, circx, :].mean() > 160 and im[circy, circx, :].mean() < 180 and ac > 0.55 ):
                image[circy, circx] = 255
                io.imsave('processing/img_ball/'+nome, (image).astype('uint8'))   
                
