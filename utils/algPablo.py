from skimage import io, color, filters
import numpy as np
from matplotlib import pyplot as plt
import os
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
import cv2

arqs = os.listdir(os.curdir)
arqs = [x for x in arqs if x.endswith('.png')]

im1 = io.imread('00249.png')
im1 = (color.rgb2gray(im1)*255).astype('uint8')

#template = im1[200:224, 342:363]
template = im1[205:219, 347:359]
w,h = template.shape

plt.figure()
for idx,a in enumerate(arqs):
    im = io.imread(a)

    Gr = (color.rgb2gray(im)*255).astype('uint8')
       
    res = cv2.matchTemplate(Gr,template,cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    seg = im.copy()
    cv2.rectangle(seg,top_left, bottom_right, 255, 2)
    
    plt.figure(1)
    plt.subplot(2,3,idx+1)
    plt.imshow(seg, cmap='gray')
    plt.title(a + '    %.4f' %max_val)
    
#    plt.figure(2)
#    plt.subplot(2,3,idx+1)
#    plt.imshow(res) 