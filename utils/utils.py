# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 21:10:29 2020

@author: felip
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from skimage import io
# models
from xgboost import XGBRegressor


import random
from matplotlib import pyplot

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_log_error


_PARAMETER_X = 4 # 4
_PARAMETER_y = 2 # 2


def metrica(y_true, y_pred):
    correct = 0
    correct_x = 0
    correct_y = 0
    for i in range(len(y_true)):
        if(y_pred[i][0] - _PARAMETER_X < y_true[i][0]  < y_pred[i][0] + _PARAMETER_X and  y_pred[i][1] - _PARAMETER_y < y_true[i][1]  < y_pred[i][1] + _PARAMETER_y ):
            correct = correct + 1
        if(y_pred[i][0] - _PARAMETER_X < y_true[i][0]  < y_pred[i][0] + _PARAMETER_X ):
            correct_x = correct_x + 1
        if(y_pred[i][1] - _PARAMETER_y < y_true[i][1]  < y_pred[i][1] + _PARAMETER_y ):
            correct_y = correct_y + 1
    print('tolerance: ' +str(_PARAMETER_X))
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
    print('mean_absolute_error: '+str((mean_absolute_error(y_true, y_pred))))
    print('mean_squared_error: '+str(np.sqrt(mean_squared_error(y_true, y_pred))))
    #print('Squared log: '+str(mean_squared_log_error(y_true, y_pred)))
    #print('Median: '+str(median_absolute_error(y_true, y_pred)))
    print('-------')

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
        

      
def plot_metros(test_video, X, y, ball_pred, save=False):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    for index, row in test_video.iterrows():
        index = int(index)
        points_pato = {'x': [], ' by':[], 'score':[]}
        ball = {'x':[], 'y':[]}
        #plt.figure() 
        
        rect = patches.Rectangle((-1, -1), 42, 22, linewidth=0.1,
                             edgecolor='r', facecolor='darkgreen', zorder=0)

        fig, ax = plt.subplots(1)
        ax.add_patch(rect)
        # Main pitch markings, ie sidelines, penalty area and halfway line
        plt.plot([0, 0,  0, 40, 40,0,0,20,20,40], 
                 [0, 0, 20, 20, 0,0,20,20,0,0], color='white')
        
        plt.plot([33.5, 34],[10, 10], color='white')
        plt.plot([6, 6.5],[10, 10], color='white')
        plt.plot([19.75, 20.25],[10, 10], color='white')
        
        centre_circle = patches.Circle([20, 10], 2.99, edgecolor='white', facecolor='darkgreen')
        ax.add_patch(centre_circle)
        
        left_arc = patches.Arc([0, 10], 12.5, 15, theta1=270.0, theta2=90.0, color='white')
        ax.add_patch(left_arc)
        
        right_arc = patches.Arc([40, 10], 12.5, 15, theta1=90.0, theta2=270.0, color='white')
        ax.add_patch(right_arc)
    
        plt.axis('off')   
        img = row[515]
        plt.title(img)
        #plt.plot(pred[index][0], pred[index][1], 'b*') 
        #plt.text(pred[index][0]-5, pred[index][1]-10, ' predicted ball' )
        plt.text(50, 50, 'Qt. Jogadores:'+ str((pd.isna(row[0:510]).value_counts()/51)[0] ), fontsize=15, color='red')
        
        points_pato['x'] = [X[index][i] for i in range(0, 80, 2)] 
        points_pato['y'] = [X[index][i] for i in range(1, 80, 2)] 
        
        ball['x'] = y[index][0]
        ball['y'] = y[index][1]
        print(ball)
        plt.plot(points_pato['x'], points_pato['y'], 'ro')  
        plt.plot(ball['x'], ball['y'], 'yx') 
        '''plt.text(ball['x'], ball['y'], 'real ball' )'''
        
        if(len(ball_pred)):
            plt.plot(ball_pred[index,0], ball_pred[index,1], 'b*') 
            '''plt.text(ball_pred[index,0], ball_pred[index:,1], 'tracked ball' )'''

        plt.xlim(0, 45)
        plt.ylim(25, 0)
        #plt.show() 
        plt.title(img) 
        
        plt.pause(1)
        save = 'out/videos/temp/'+str(img)
        plt.savefig(save, format='png')   
        #plt.clf()
    plt.close('all')
    

def convert_to_np(data):
    label_pes = pd.read_csv("in/csv/label_pes.csv")
    label_pes = list(label_pes.columns[:])
    data = data.drop(label_pes, axis=1 , errors='ignore')
    
    # usar -4 - time com posse
    # usar -5 - sem time com posse
    
    # Utilizando uma unica variavel de treinamento 
    
    _INDEX_V = len(data.columns) - 5
    _INDEX_X = len(data.columns) - 4
    _INDEX_Y = len(data.columns) - 2
    
    X = data.iloc[:, 0:_INDEX_V].values
    y = data.iloc[:, _INDEX_X:_INDEX_Y].values
    #y = y.astype(int)
    index = data.index.values
    return X,y,index
        
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

def find_the_ball_CCOEFF_NORMED(nome):
    import cv2
    _MAX_STD = 5
    _MIN_SCORE = 0.85
    
    pasta = 'processing/img_recortadas/'
    im = io.imread(pasta + nome)
    
    
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
    
    desvio_x = np.array([max_loc1[0], max_loc2[0], max_loc3[0]]).std()
    desvio_y = np.array([max_loc1[1], max_loc2[1], max_loc3[1]]).std()
    media_score = np.array([max_val1, max_val2, max_val3]).mean()
    
    if(desvio_x < _MAX_STD and desvio_y < _MAX_STD and media_score > _MIN_SCORE):
        return max_loc1[1], max_loc1[0]
    else:
        return -1,-1

from datetime import datetime

def save_np(test, pred, file=datetime.now().strftime('%d_%m_%Y_%H_%M') ):
    name=file+'.npy'
    path='processing/numpy/'
    with open(path+name, 'wb') as f:
        np.save(f, test)
        np.save(f, pred)

    return {'name': name, 'path': path}
    