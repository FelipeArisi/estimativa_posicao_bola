# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 21:08:28 2020

@author: felip
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
# models
from sklearn.model_selection import KFold
from xgboost import XGBRegressor



from utils.utils import criar_dados, print_error, metrica, shuffle, find_the_ball_CCOEFF_NORMED, convert_to_np, plot_metros
from utils.frame_to_video import frame_to_video 

from utils.correcao import correcao, correcao_data, correcao_img
from utils.persp2 import persp2, persp2_data, persp2_img
from utils.erro_euclideano import erro_euclideano, erro_euclideano_data
data = pd.read_csv("in/csv/data_tcc_20_07.csv")
data_test = pd.read_csv("in/csv/data_teste_20_07.csv")
_PARAMETER = 2


#%% 
# UTLIZADO QUANTO É APENAS UM TREINAMENTO 
   
X,y,index = convert_to_np(data)
model_x = XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1009,
                max_depth = 25, alpha = 10, n_estimators = 500)
model_y = XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
               max_depth = 25, alpha = 10, n_estimators = 1000)

model_x = XGBRegressor(learning_rate = 0.1)
model_y = XGBRegressor(learning_rate = 0.1)
 
#from sklearn.neural_network import MLPRegressor
#model_x =  MLPRegressor(learning_rate = 'adaptive', max_iter=5000, activation='logistic')
#model_y =  MLPRegressor(learning_rate = 'adaptive', max_iter=5000, activation='logistic')


#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
#kernel = DotProduct() + WhiteKernel()
#model_x =  GaussianProcessRegressor(kernel=kernel,random_state=0)
#model_y =  GaussianProcessRegressor(kernel=kernel,random_state=0)


#from sklearn.ensemble import RandomForestRegressor
#model_x = RandomForestRegressor()
#model_y = RandomForestRegressor()

#X,y = criar_dados(100,X,y)
X = np.load("novo_treinamento_X.npy")
y = np.load("novo_treinamento_y.npy")
#kf = KFold(n_splits=2)
y_pred = []

_i = 0 

test = np.array([], dtype=np.int64).reshape(0,2)
pred = np.array([], dtype=np.int64).reshape(0,2)


## USAR SEM O KFOLD
#X = shuffle(X)
#save_csv(X)
indices = np.arange(X.shape[0])
rng = np.random.RandomState(123)
permuted_indices = rng.permutation(indices)

## seperar a unica base para treino e teste
train_size, valid_size = int(0.95*X.shape[0]), int(0.05*X.shape[0])
train_ind = permuted_indices[:train_size]
valid_ind = permuted_indices[train_size:(train_size + valid_size)]

X_train, y_train = X[train_ind], y[train_ind]
X_test, y_test = X[valid_ind], y[valid_ind]

'''
## utilizar um outro arquivo para o teste
X_train, y_train = X,y
X_test, y_test, index_test = convert_to_np(data_test)
#X_train,y_train = criar_dados(1000, X_train,y_train)
#X_test,y_test = criar_dados(200,X_test, y_test)

### Converter Para metros 
X_train, y_train = persp2_data( correcao_data(X_train) ), persp2_data (correcao_data(y_train)) 
X_test, y_test = persp2_data (correcao_data(X_test) ), persp2_data( correcao_data(y_test))'''

#X_train, y_train =  correcao_data(X_train) , correcao_data(y_train)
#X_test, y_test = correcao_data(X_test) , correcao_data(y_test)

model_x.fit(X_train,y_train[:,0])
_y_pred_x = model_x.predict(X_test)

model_y.fit(X_train, y_train[:,1])
_y_pred_y = model_y.predict(X_test)

y_pred = np.vstack((_y_pred_x, _y_pred_y))
y_pred = y_pred.T

test = np.vstack((test,y_test))
pred = np.vstack((pred,y_pred))
###
'''
# cross validation

for index_train, index_test in kf.split(X):
    X_train, X_test, y_train, y_test = X[index_train], X[index_test], y[index_train], y[index_test]

    model_x.fit(X_train,y_train[:,0])
    _y_pred_x = model_x.predict(X_test)
    
    model_y.fit(X_train, y_train[:,1])
    _y_pred_y = model_y.predict(X_test)
    
    y_pred = np.vstack((_y_pred_x, _y_pred_y))
    y_pred = y_pred.T
  
    
    test = np.vstack((test,y_test))
    pred = np.vstack((pred,y_pred))

#plot_ball(test, pred)
#print_importances(importances_x, importances_y)
'''
print_error(test, pred)
metrica(test, pred)


#%% Testar com videos -- Correção
## Aqui será treinado um frame por vez e utilizado a respota para o frame seguinte 

base_name = 'ataquepato'
test_video = pd.read_csv("in/csv/test/"+base_name+".csv")

X,y,index = convert_to_np(test_video)
test = y
X = persp2_data (correcao_data(X) )
test= persp2_data (correcao_data(y.astype(float)))

aux = np.zeros((1,82))

pred = np.array([], dtype=np.int64).reshape(0,2)
#test = np.array([], dtype=np.float).reshape(0,2)

#test = test.astype(float)
for i in index:
    aux[0] = X[i]
    x = model_x.predict(aux)
    y = model_y.predict(aux)
    _nameimg = test_video['img'][i]
    im = io.imread('in/img/'+_nameimg)
    #im = persp2_img('in/img/'+_nameimg)
    aux[0] = X[i]
    
    im_ball = im
    im_ball = im[int(y-225):int(y+225),int(x-275):int(x+275),:]  

    
    # recortar a quadra inteira
    im_ball = im[180:820, :]

    
    io.imsave('processing/img_recortadas/'+_nameimg, (im_ball).astype('uint8'))
    
    find_y, find_x = find_the_ball_CCOEFF_NORMED(_nameimg)
    #find_y = -1
    #find_x = -1
    print(find_y)
    if(find_x == -1):
        _y_pred_y, _y_pred_x = y, x
    else:
        _y_pred_y = y - im_ball.shape[0]/2 + find_y
        _y_pred_x = x - im_ball.shape[1]/2 + find_x
        #_y_pred_y = find_y
        #_y_pred_x = find_x
    
    #im_ball = im[int(_y_pred_y-150):int(_y_pred_y+150),int(_y_pred_x-200):int(_y_pred_x+200),:]    
    #io.imsave('img_recortadas/'+_nameimg, (im_ball).astype('uint8'))
    
    
    if((i+1) != len(index)):
        X[i+1,80] = _y_pred_x
        X[i+1,81] = _y_pred_y
    y_pred = np.vstack((_y_pred_x, _y_pred_y))
    y_pred = y_pred.T
    pred = np.vstack((pred,y_pred))


#%% Testar com videos -- Metros
## Aqui será treinado um frame por vez e utilizado a respota para o frame seguinte 

base_name = 'ataquepato'
test_video = pd.read_csv("in/csv/test/"+base_name+".csv")

X,y,index = convert_to_np(test_video)
test = y
X = persp2_data (correcao_data(X) )
test= persp2_data (correcao_data(y.astype(float)))

aux = np.zeros((1,82))

pred = np.array([], dtype=np.int64).reshape(0,2)
test = test.astype(float)
for i in index:
    
    aux[0] = X[i]
    x = model_x.predict(aux)
    y = model_y.predict(aux)
    _nameimg = test_video['img'][i]
    
    _y_pred_y, _y_pred_x = y, x

    
    if((i+1) != len(index)):
        X[i+1,80] = _y_pred_x
        X[i+1,81] = _y_pred_y
    y_pred = np.vstack((_y_pred_x, _y_pred_y))
    y_pred = y_pred.T
    pred = np.vstack((pred,y_pred))
    
print_error(test, pred)
metrica(test, pred)


#%%

plot_metros(test_video, X , test, pred, save=True)
#%% PLOTAR JOGADORES    
for index, row in test_video.iterrows():
    print (row[515])
    #plt.figure() 
    img = row[515]
    #index = row[84]
    im = correcao_img('in/img/' + img)
    #plt.imshow(io.imread('in/img/' + img))
    #plt.imshow(im)
    #plt.title(img)
    plt.plot(pred[index][0], pred[index][1], 'b*') 
    plt.text(pred[index][0]-5, pred[index][1]-10, ' predicted ball' )
    plt.text(50, 50, 'Qt. Jogadores:'+ str((pd.isna(row[0:510]).value_counts()/51)[0] ), fontsize=15, color='red')
    plt.xlim(0, 40)
    plt.ylim(20, 0)
    plt.show() 
    plt.title(img) 
    
    plt.pause(1)
   
    save = 'out/videos/temp/'+str(img)
    #plt.savefig(save, format='png')   
    plt.clf()
plt.close('all')

#%%
frame_to_video('out/videos/temp/', 'meiocampometros.mp4')

#%% savar in np 
save =  np.concatenate((test, pred), axis=1)

with open('processing/numpy/resultado_corr_test.npy', 'wb') as f:
    np.save(f, test)
    np.save(f, pred)
    
#%% 
import cv2
correc = correcao('resultado_corr.npy', 'processing/numpy/')

#%%

persp = persp2('resultado_corr_test.npy', 'processing/numpy/')

#%%

erro_euclideano(persp['name'], persp['path'])