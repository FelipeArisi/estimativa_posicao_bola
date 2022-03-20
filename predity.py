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

from sklearn.impute import SimpleImputer

from utils.utils import criar_dados, print_error, metrica, shuffle, find_the_ball_CCOEFF_NORMED, convert_to_np, plot_metros
from utils.frame_to_video import frame_to_video 

from utils.correcao import correcao, correcao_data, correcao_img
from utils.persp2 import persp2, persp2_data, persp2_img
from utils.erro_euclideano import erro_euclideano, erro_euclideano_data
from utils.replaceNan import replaceNan
data = pd.read_csv("in/csv/data_tcc_20_07.csv")
data_test = pd.read_csv("in/csv/data_teste_20_07.csv")
_PARAMETER = 2
from numba import cuda


#%% 
# UTLIZADO QUANTO É APENAS UM TREINAMENTO 

data = pd.read_csv("in/csv/data_tcc_20_07.csv")
parametros = {'colsample_bytree': 0.8, 'max_depth': 15, 'n_estimators': 1000, 'reg_alpha': 1.3, 'reg_lambda': 1.3, 'subsample': 0.9}
X,y,index = convert_to_np(data)
model_x = XGBRegressor()
model_y = XGBRegressor()

#imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
#imp.fit(X)
#X = imp.transform(X)

X,y = criar_dados(30000,X,y)
y_pred = []

_i = 0 

test = np.array([], dtype=np.int64).reshape(0,2)
pred = np.array([], dtype=np.int64).reshape(0,2)

## utilizar um outro arquivo para o teste
X_train, y_train = X,y
X_test, y_test, index_test = convert_to_np(data_test)
#imp.fit(X_test)
#X_test = imp.transform(X_test)
#X_train,y_train = criar_dados(1000, X_train,y_train)
#X_test,y_test = criar_dados(200,X_test, y_test)


X_train, y_train = persp2_data (correcao_data(X_train) ), persp2_data (correcao_data(y_train)) 
X_test, y_test  =   persp2_data (correcao_data(X_test) ), persp2_data( correcao_data(y_test))

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

print_error(test, pred)
metrica(test, pred)

#%% MultiOutputRegressor

from sklearn.multioutput import MultiOutputRegressor
model = XGBRegressor()

multioutputregressor = MultiOutputRegressor(model).fit(X, y)
multioutputregressor.predict(X)
print (np.mean((multioutputregressor.predict(X) - y)**2, axis=0))
#%%
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor

X,y,index = convert_to_np(data)
X,y = criar_dados(30000,X,y)

#imp = SimpleImputer(missing_values=np.nan, strategy='mean')
#imp.fit(X)
#X = imp.transform(X)

X, y = persp2_data( correcao_data( X)), persp2_data( correcao_data(y))


def algorithm_pipeline(X_train_data, y_train_data, 
                       model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
                       do_probabilities = False):
    print('Entrou')
    

    return fitted_model


model = XGBRegressor()
param_grid = {
    'estimator__n_estimators': [400, 600, 800],
    'estimator__colsample_bytree': [0.7, 0.6, 0.8],
    'estimator__max_depth': [15,20,25],
    'estimator__reg_alpha': [1.1,1.2, 1.3],
    'estimator__reg_lambda': [1.1, 1.3],
    'estimator__subsample': [0.7,0.8, 0.9],
    'estimator__min_child_weight': [1, 5, 10],
    'estimator__learning_rate': [0.01, 0.02, 0.05]  
    
}

gs = RandomizedSearchCV(
        estimator = MultiOutputRegressor(model),
        param_distributions =param_grid, 
        cv=5, 
        n_jobs=-1, 
        scoring='neg_mean_squared_error',
        verbose=3,
        n_iter=10,
        return_train_score=True
    )
fitted_model = gs.fit(X, y)
    


# Root Mean Squared Error
print(np.sqrt(-fitted_model.best_score_))
print(fitted_model.best_params_)


#%%

resultTrain = fitted_model.predict(X)


matriz = np.concatenate((resultTrain, y), axis=1)

resultTrain = fitted_model.predict(X_test)
matriz2 = np.concatenate((resultTrain, y_test), axis=1)


erro_euclideano_data(matriz, matriz2)

#%% Print error Grid
model_x =  MultiOutputRegressor(XGBRegressor())
model_x.fit(X, y)

#%%
print("Print erro treino")
y_treino = model_x.predict(X)
y_search = fitted_model.predict(X)

print_error(y, y_treino)
print("-grid")
print_error(y, y_search)

print("Print Teste")
y_treino = model_x.predict(X_test)
y_search = fitted_model.predict(X_test)


print_error(y_test, y_treino)
print("-grid")
print_error(y_test, y_search)



#%% Mapa de calor

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
X,y,index = convert_to_np(data)

fig, axs = plt.subplots(2)
plt.subplot(1, 3, 1)
plt.hist2d(y[:,0],y[:,1],bins=25)

y=  correcao_data(y)
plt.subplot(1, 3, 2)
plt.hist2d(y[:,0],y[:,1],bins=25)


plt.show()

y= persp2_data(y)
plt.subplot(1, 3, 3)
plt.hist2d(y[:,0],y[:,1],bins=25)


plt.show()


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