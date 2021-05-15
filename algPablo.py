from skimage import io, color, filters
import numpy as np
from matplotlib import pyplot as plt
import os
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
import cv2

path = 'processing/img_recortadas/ataquepato/'
#arqs = os.listdir(os.curdir)
arqs = os.listdir(path)
arqs = [x for x in arqs if x.endswith('.png')]

im1 = io.imread('processing/temp/00249.png')
im1 = (color.rgb2gray(im1)*255).astype('uint8')


im2 = io.imread('in/img/00380.png')
im2 = (color.rgb2gray(im2)*255).astype('uint8')

im3 = io.imread('in/img/00939.png')
im3 = (color.rgb2gray(im3)*255).astype('uint8')


template1 = im1[200:219, 205:220]
template2 = im2[303:314, 1115:1127]
template3 = im3[257:266, 515:524]
#template = im1[205:219, 347:359]
w,h = template1.shape


#%%

#plt.figure()
with open("processing/temp/result_img3/ataquepato.txt", "w") as txt_file:
    for idx,a in enumerate(arqs):
        im = io.imread(path+a)
    
        Gr = (color.rgb2gray(im)*255).astype('uint8')
           
        res1 = cv2.matchTemplate(Gr,template1, cv2.TM_CCOEFF_NORMED)
        min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
        
        res2 = cv2.matchTemplate(Gr,template2, cv2.TM_CCOEFF_NORMED)
        min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
        
        res3 = cv2.matchTemplate(Gr,template3, cv2.TM_CCOEFF_NORMED)
        min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(res3)
    
        top_left1 = max_loc1
        bottom_right1 = (top_left1[0] + w, top_left1[1] + h)
        
        top_left2 = max_loc2
        bottom_right2 = (top_left2[0] + w, top_left2[1] + h)
        
        top_left3 = max_loc3
        bottom_right3 = (top_left3[0] + w, top_left3[1] + h)
        
        seg = im.copy()
        cv2.rectangle(seg,top_left1, bottom_right1, (255,0,0), 2)
        cv2.rectangle(seg,top_left2, bottom_right2, (0,255,0), 2)
        cv2.rectangle(seg,top_left3, bottom_right3, (0,0,255), 2)
        
        #plt.figure(1)
        #plt.subplot(2,3,idx+1)
        plt.imshow(seg, cmap='gray')
        plt.title(a + '    %.2f' %max_val1 + '    %.2f' %max_val2+ '    %.2f' %max_val3)
        save = 'processing/temp/result_img3/'+a+'.png'
        plt.savefig(save)
        
        mediax = round(np.array([max_loc1[0], max_loc2[0], max_loc3[0]]).mean(),2)
        mediay = round(np.array([max_loc1[1], max_loc2[1], max_loc3[1]]).mean(),2)
        desvio = round(np.array([max_loc1, max_loc2, max_loc3]).std(),2)
        
        #texto = str(max_loc1)+"/"+str(max_val1)+"/"+str(max_loc2)+"/"+str(max_val2)+"/"+str(max_loc3)+"/"+str(max_val3)+"/("+str(mediax)+", "+str(mediay)+")/"+str(desvio)+"/"+a+"\n"
        texto = str(max_loc1)+"/"+str(max_val1)+"/"+str(max_loc2)+"/"+str(max_val2)+"/"+str(max_loc3)+"/"+str(max_val3)+"/ "+a+"\n"
        txt_file.write(texto)
    