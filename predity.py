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

from scipy import stats
import random
from matplotlib import pyplot

from utils.utils import criar_dados, print_error, metrica, shuffle, find_the_ball

data = pd.read_csv("csv_files/data_tcc_22_08.csv")
_PARAMETER = 100


#%%

def convert_to_np(data):
    label_pes = pd.read_csv("csv_files/label_pes.csv")
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
#%% 
# UTLIZADO QUANTO É APENAS UM TREINAMENTO 
   
X,y,index = convert_to_np(data)
model_x = XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1009,
                max_depth = 25, alpha = 10, n_estimators = 500)
model_y = XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 16, alpha = 10, n_estimators = 1000)


X,y = criar_dados(10000,X,y)
kf = KFold(n_splits=2)
y_pred = []

_i = 0 

test = np.array([], dtype=np.int64).reshape(0,2)
pred = np.array([], dtype=np.int64).reshape(0,2)


## USAR SEM O KFOLD
X = shuffle(X)
#save_csv(X)
indices = np.arange(X.shape[0])
rng = np.random.RandomState(123)
permuted_indices = rng.permutation(indices)

train_size, valid_size = int(0.95*X.shape[0]), int(0.05*X.shape[0])
train_ind = permuted_indices[:train_size]
valid_ind = permuted_indices[train_size:(train_size + valid_size)]

X_train, y_train = X[train_ind], y[train_ind]
X_test, y_test = X[valid_ind], y[valid_ind]

#X_train,y_train = criar_dados(1000, X_train,y_train)
#X_test,y_test = criar_dados(200,X_test, y_test)

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

#%%
# REALIZAR DOIS TREINOS UM COM TODOS OS JOGADORES DETECTADOS PELO ALPHA POSE OUTRO NÃO

X,y,index = convert_to_np(data)

# pega somente os 96 primeiros itens(onde tem todo os jogadores)
# pega somente os pontos dos jogadores (não usa a bola antiga)
X_parcial = X[0:96,0:80].copy()
y_parcial = y[0:96:].copy()

X,y = criar_dados(10000,X,y)
X_parcial,y_parcial = criar_dados(10000,X_parcial,y_parcial)

model_x = XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1009,
                max_depth = 25, alpha = 10, n_estimators = 500)
model_y = XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 16, alpha = 10, n_estimators = 1000)

#PARCIAL É QUANTO TEM OS 10 JOGADORES 
model_parcial_x = XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1009,
                max_depth = 25, alpha = 10, n_estimators = 1000)
model_parcial_y = XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 16, alpha = 10, n_estimators = 1000)

## COMO FICARIA DIFICIL VER A METRICA, ACABEI USANDO TODOS OS DADOS PARA OS TREINAMENTOS 
model_x.fit(X,y[:,0])
model_y.fit(X,y[:,1])

model_parcial_x.fit(X_parcial,y_parcial[:,0])
model_parcial_y.fit(X_parcial,y_parcial[:,1])

#%% UTILIZADO COM OS DOIS TREINAMENTOS 
# se achar todos os jogadores utilizar o model_parcial

test_video = pd.read_csv("csv_files/test/ataquepato.csv")

X,y,index = convert_to_np(test_video)

pred = np.array([], dtype=np.int64).reshape(0,2)
for i in index:
    
    if np.sum(np.isnan(X[i]))/8 != 0 : # se tiver todos os jogadores muda o treinamento 
        print('Treino 1')
        _y_pred_x = model_x.predict(X[i:i+1,:])
        _y_pred_y = model_y.predict(X[i:i+1,:])
    else:
        print('Treino 2')
        _y_pred_x = model_parcial_x.predict(X[i:i+1,0:80])
        _y_pred_y = model_parcial_y.predict(X[i:i+1,0:80])
        
    if((i+1) != len(index)):
        X[i+1,80] = _y_pred_x
        X[i+1,81] = _y_pred_y
    y_pred = np.vstack((_y_pred_x, _y_pred_y))
    y_pred = y_pred.T
    pred = np.vstack((pred,y_pred))
    
    

#%% Testar com videos
## Aqui será treinado um frame por vez e utilizado a respota para o frame seguinte 

test_video = pd.read_csv("csv_files/test/ataquepato.csv")

X,y,index = convert_to_np(test_video)

aux = test = np.zeros((1,82))
pred = np.array([], dtype=np.int64).reshape(0,2)

for i in index:
    _nameimg = test_video['img'][i]
    im = io.imread('img/'+_nameimg)
    
    aux[0] = X[i]
    x = model_x.predict(aux)
    y = model_y.predict(aux)
    im_ball = im[int(y-150):int(y+150),int(x-200):int(x+200),:]  
    
    io.imsave('img_recortadas/'+_nameimg, (im_ball).astype('uint8'))
    
    #find_y, find_x = find_the_ball(_nameimg)
    find_y = -1
    find_x = -1
    print(find_y)
    if(find_x == -1):
        _y_pred_y, _y_pred_x = y, x
    else:
        _y_pred_y = y - im_ball.shape[0]/2 + find_y
        _y_pred_x = x - im_ball.shape[1]/2 + find_x
    
    #im_ball = im[int(_y_pred_y-150):int(_y_pred_y+150),int(_y_pred_x-200):int(_y_pred_x+200),:]    
    #io.imsave('img_recortadas/'+_nameimg, (im_ball).astype('uint8'))
    
    
    if((i+1) != len(index)):
        X[i+1,80] = _y_pred_x
        X[i+1,81] = _y_pred_y
    y_pred = np.vstack((_y_pred_x, _y_pred_y))
    y_pred = y_pred.T
    pred = np.vstack((pred,y_pred))

#%% PLOTAR JOGADORES    
for index, row in test_video.iterrows():
    print (row[515])
    #plt.figure() 
    img = row[515]
    #index = row[84]
    plt.imshow(io.imread('img/' + img))
    plt.title(img)
    plt.plot(pred[index][0], pred[index][1], 'b*') 
    plt.text(pred[index][0]-5, pred[index][1]-10, ' predicted ball' )
    plt.text(50, 50, 'Qt. Jogadores:'+ str((pd.isna(row[0:510]).value_counts()/51)[0] ), fontsize=15, color='red')
    plt.xlim(0, 1920)
    plt.ylim(1080, 0)
    plt.show() 
    plt.title(img) 
    
    plt.pause(1)
    plt.clf()
    save = 'videos_2/data_tcc_ataquepato/'+str(img)
    #plt.savefig(save, format='png')   

plt.close('all')
#%% savar in np 
save =  np.concatenate((test, pred), axis=1)

with open('np_files/result_pixel_22_08_2.npy', 'wb') as f:
    np.save(f, test)
    np.save(f, pred)


    

