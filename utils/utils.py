# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 21:10:29 2020

@author: felip
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
# models
from xgboost import XGBRegressor


import random
from matplotlib import pyplot

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error


_PARAMETER = 100

def metrica(y_true, y_pred):
    correct = 0
    correct_x = 0
    correct_y = 0
    for i in range(len(y_true)):
        if(y_pred[i][0] - _PARAMETER < y_true[i][0]  < y_pred[i][0] + _PARAMETER and  y_pred[i][1] - _PARAMETER < y_true[i][1]  < y_pred[i][1] + _PARAMETER ):
            correct = correct + 1
        if(y_pred[i][0] - _PARAMETER < y_true[i][0]  < y_pred[i][0] + _PARAMETER ):
            correct_x = correct_x + 1
        if(y_pred[i][1] - _PARAMETER < y_true[i][1]  < y_pred[i][1] + _PARAMETER ):
            correct_y = correct_y + 1
    print('tolerance: '+str(_PARAMETER))
    print('correct: '+str(correct*100/len(y_true)))
    print('correct_x: '+str(correct_x*100/len(y_true)))
    print('correct_y: '+str(correct_y*100/len(y_true)))
    
def axis(keypoints):
    points = {'x': [], 'y':[], 'score':[]}
    points['x'] = [keypoints[i] for i in range(0, 510, 3)] 
    points['y'] = [keypoints[i] for i in range(1, 510, 3)] 
    points['score'] = [keypoints[i] for i in range(2, 510, 3)] 
    return points

def print_error(y_true, y_pred):
    print('Squared: '+str(np.sqrt(mean_squared_error(y_true, y_pred))))
    print('Squared log: '+str(mean_squared_log_error(y_true, y_pred)))
    #print('Median: '+str(median_absolute_error(y_true, y_pred)))
    print('-------')
    return np.sqrt(mean_squared_error(y_true, y_pred))

def print_img_train(y_pred,index_t):
    for i in range(len(index_t)):
        ball_pred = {'x':[], 'y':[]}
        ball_pred['x'] = y_pred[i][0]
        ball_pred['y'] = y_pred[i][1]
        print_img( int(index_t[i]), int(index_t[i]) + 1, ball_pred  )
    

def print_img(ini, fim, ball_pred):
    for index, row in sorted(data.iterrows())[ini:fim]:
        plt.figure()
        points_pato = {'x': [], ' by':[], 'score':[]}
        points_visitante = {'x': [], 'y':[], 'score':[]}
        ball = {'x':[], 'y':[]}
        img = row[513]
        index = row[514]
        if(index < 50 or index > 96):
            plt.imshow(io.imread('in/img/' + img))
        else:
            plt.imshow(io.imread('in/img_1/' + img))
        points_pato['x'] = [row[i] for i in range(0, 255, 3)] 
        points_pato['y'] = [row[i] for i in range(1, 255, 3)] 
        points_pato['score'] = [row[i] for i in range(2, 255, 3)] 
        
        points_visitante['x'] = [row[i] for i in range(255, 510, 3)] 
        points_visitante['y'] = [row[i] for i in range(256, 510, 3)] 
        points_visitante['score'] = [row[i] for i in range(257, 510, 3)] 
        
        ball['x'] = row[511]
        ball['y'] = row[512]
        
        plt.plot(points_pato['x'], points_pato['y'], 'ro') 
        plt.plot(points_visitante['x'], points_visitante['y'], 'go') 
        plt.plot(ball['x'], ball['y'], 'yx') 
        plt.text(ball['x']-5, ball['y']-10, 'real ball' )
        
        if(len(ball_pred)):
            plt.plot(ball_pred['x'], ball_pred['y'], 'b*') 
            plt.text(ball_pred['x']-5, ball_pred['y']-10, 'tracked ball' )
        
        plt.xlim(0, 1920)
        plt.ylim(1080, 0)
        plt.show() 
        plt.title(img) 
        save = str(img)+'.png'
        plt.savefig(save, format='png')
        
def searchCV(X_train, y_train):
    params = {  
          'colsample_bytree':[i/10.0 for i in range(1,7)],
          'learning_rate': [i/10.0 for i in range(1,7)],
          'max_depth': [5,10,20,30],
          'n_estimators': [2500,1000,3000,4000]}

    xgb = XGBRegressor(nthread=-1)
    grid = GridSearchCV(xgb, params)
    grid.fit(X_train,y_train[:,0])
    
def plot_ball(y_test, y_pred):    
    plt.figure()    
    for i in range(len(y_test)):
        ball = np.zeros((2,2), dtype=np.float64)
        ball[0,0] = y_test[i,0]
        ball[0,1] = y_test[i,1]
        ball[1,0] = y_pred[i,0]
        ball[1,1] = y_pred[i,1]
        plt.plot(ball[0:,0], ball[0:,1], 'go') 
        plt.plot(ball[1:,0], ball[1:,1], 'ro')
        plt.plot(ball[:,0], ball[:,1], 'k-')
    plt.xlim(0, 1920)
    plt.ylim(1080, 0)
    plt.show() 
    plt.title('Result') 

       
def return_importances(model, plt):
    importance = model.feature_importances_
    # summarize feature importance
    # plot feature importance
    if(plt):
        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.legend()
        pyplot.show()
    return importance

def shuffle(X):
    posicoes = np.arange(0,10)
    random.shuffle(posicoes) #embaralha posicoes
    novo = np.zeros(X.shape)
    if(X.shape[1] == 82): #CUIDA QUANDO USA A BOLA ANTERIOR OU NÃO
        novo[:,80:82] = X[:,80:82].copy()
    for n, pos in enumerate(posicoes):
        pos_velha = n*8
        pos_nova = pos*8
        novo[:,pos_nova:pos_nova+8] = X[:,pos_velha:pos_velha+8].copy()
        
    return novo

# funcao para criar novos dados mas mudando a ordem dos jogadores
def criar_dados(n_dados, X, y):
    i = 0
    while i < n_dados: # cria quantos jogadores for necessario
        index = (np.random.choice(len(X), 1))[0] # pega um frame aleatorio
        X = np.vstack((X,shuffle(X[index:index+1,:]))) # troca a posição e salva
        y = np.vstack((y,y[index])) # salva a posição da bola naquele frame
        i=i+1
    return X,y

def save_csv(data, folder):
    # save numpy array as csv file
    from numpy import savetxt
    # define data
    # save to csv file
    savetxt(folder, data, delimiter=',')

def print_importances(importances_x, importances_y): 
    labels = data.columns[:]
    labels = labels[:-5]
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    ax.bar(x - width/2, importances_x, width, label='importances_x')
    ax.bar(x + width/2, importances_y, width, label='importances_y')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Importances')
    ax.set_ylabel('Ponto do jogador')
    ax.set_title('Importances X e Y do treinamento')
    #ax.set_xticks(x)
    #ax.set_xticklabels(labels)
    ax.legend()  
    
    fig.tight_layout()
    
    plt.show()

from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter, circle
from skimage.util import img_as_ubyte
from skimage import data, color
'''
def ball(mean, accums):
    i = 0
    for m, a in zip(mean, accums) :
         if( ( (m > 150.0 and m < 190.0) and a > 0.50 ) or a > 0.8 ):
             return i
         i = i+1
    return -1    

def find_the_ball(nome):
    pasta = 'img_recortadas/'
    im = io.imread(pasta + nome)
    # Load picture and detect edges
    image = img_as_ubyte(im[:,:,0])
    edges = canny(image, sigma=3, low_threshold=75, high_threshold=85)
    
    # Detect two radii
    hough_radii = np.arange(3, 15, 1)
    hough_res = hough_circle(edges, hough_radii)
    
    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=3)
    mean = np.mean(im[cy[:],cx[:],], axis=1)
    # Draw them
    
    
    _id  = ball(mean, accums)
    #_id = 2
    if(_id != -1):
        accums, center_x, center_y, radius = accums[_id], cx[_id], cy[_id], radii[_id]
        #fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
        image = color.gray2rgb(image)
        circy, circx = circle_perimeter(center_y, center_x, radius,
                                        shape=image.shape)
        image[circy, circx] = (220, 20, 20)
        io.imsave('img_ball/'+nome, (image).astype('uint8'))
        return center_y, center_x
    else:
        return -1,-1'''
from skimage.filters.rank import median
from skimage.morphology import disk

def find_the_ball(nome):        
    pasta = 'processing/img_recortadas/'
    im = io.imread(pasta + nome)
    
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

    # Load picture and detect edges
    image = img_as_ubyte(im[:,:,0])
    edges = canny(image, sigma=3, low_threshold=30, high_threshold=85)
    
    # Detect two radii
    hough_radii = np.arange(3, 15, 1)
    hough_res = hough_circle(edges, hough_radii)
    
    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=3)
    #mean = np.mean(im[cy[:],cx[:],], axis=1)
    # Draw them
    
    
    #_id  = ball(mean, accums)
    #if(_id != -1):
    #accums, center_x, center_y, radius = accums[_id], cx[_id], cy[_id], radii[_id]
    for center_y, center_x, radius, ac in zip(cy, cx, radii, accums):
        #accums, center_x, center_y, radius = accums[0], cx[0], cy[0], radii[0]
        image = color.gray2rgb(image)
        circy, circx = circle(center_y, center_x, radius,
                                        shape=image.shape)
        #image[circy, circx] = 255
        image = im
        if(image[circy, circx, :].mean() > 160 and image[circy, circx, :].mean() < 180 and ac > 0.55 and np.std(image[circy, circx, :]) > 20 and np.std(image[circy, circx, :]) < 45 ):
            print(nome +' - accums: ' +str(round(ac,2)) + ' - mean: '+ str(round(image[circy, circx, :].mean(),2)))
            image[circy, circx] = 255
            io.imsave('img_ball/'+nome, (image).astype('uint8'))
            #plt.imshow(image, cmap=plt.cm.gray)
            #plt.show()
            circy, circx = circle_perimeter(center_y, center_x, radius,
                                        shape=image.shape)
            return center_y, center_x
    return -1,-1
