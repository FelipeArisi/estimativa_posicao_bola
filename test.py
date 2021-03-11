# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:31:29 2020

@author: felipe
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
# models
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from scipy import stats
import random

from matplotlib import pyplot

#data = pd.read_csv("csv_files/data_tcc_test.csv")
data = pd.read_csv("in/csv/data_tcc_21_08.csv")
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
            plt.imshow(io.imread('img/' + img))
        else:
            plt.imshow(io.imread('img_1/' + img))
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
        
def searchCV():
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
    novo = np.zeros((len(X),80))
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



#%%
# usando arquivo sem score    
"""
label_score = pd.read_csv("csv_files/label_score.csv")
label_score = list(label_score.columns[:])
data = data.drop(label_score, axis=1 , errors='ignore')

label_goalkeeper = pd.read_csv("csv_files/label_goalkeeper.csv")
label_goalkeeper = list(label_goalkeeper.columns[:])
data = data.drop(label_goalkeeper, axis=1 , errors='ignore')


label_eyes = pd.read_csv("csv_files/label_eyes.csv")
label_eyes = list(label_eyes.columns[:])
data = data.drop(label_eyes, axis=1 , errors='ignore' )
"""
label_pes = pd.read_csv("csv_files/label_pes.csv")
label_pes = list(label_pes.columns[:])
data = data.drop(label_pes, axis=1 , errors='ignore')

# usar -4 - time com posse
# usar -5 - sem time com posse

# Utilizando uma unica variavel de treinamento 

_INDEX_V = len(data.columns) - 5
_INDEX_X = len(data.columns) - 4
_INDEX_Y = len(data.columns) - 2
_INDEX_ID = len(data.columns)


X = data.iloc[:, 0:_INDEX_V].values
y = data.iloc[:, _INDEX_X:_INDEX_Y].values
#y = y.astype(int)
index = data.index.values


#X = stats.zscore(X)
#%% XGBOOST
#### Utilizando o Kfold para testar todas as entradas


model_x = XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1009,
                max_depth = 25, alpha = 10, n_estimators = 500)
model_y = XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 16, alpha = 10, n_estimators = 1000)


X,y = criar_dados(10000,X,y)
kf = KFold(n_splits=12)
y_pred = []

#X = shuffle(X)
#save_csv(X)

importances_x = np.zeros(np.size(X,1))
importances_y = np.zeros(np.size(X,1))
_i = 0 

test = np.array([], dtype=np.int64).reshape(0,2)
pred = np.array([], dtype=np.int64).reshape(0,2)


for index_train, index_test in kf.split(X):
    X_train, X_test, y_train, y_test = X[index_train], X[index_test], y[index_train], y[index_test]

    model_x.fit(X_train,y_train[:,0])
    _y_pred_x = model_x.predict(X_test)
    
    model_y.fit(X_train, y_train[:,1])
    _y_pred_y = model_y.predict(X_test)
    
    y_pred = np.vstack((_y_pred_x, _y_pred_y))
    y_pred = y_pred.T
  
    importances_x = return_importances(model_x, 0) + importances_x
    importances_y = return_importances(model_y, 0) + importances_y
    _i = _i + 1
    
    test = np.vstack((test,y_test))
    pred = np.vstack((pred,y_pred))
    


#plot_ball(test, pred)
#print_importances(importances_x, importances_y)
importances_x = importances_x / _i
importances_y = importances_y / _i


print_error(test, pred)
metrica(test, pred)

#%%
##### CRIANDO NOVOS DADOS #######
## Deixar separado os novos dados
'''X = pd.read_csv('csv_files/X.csv', header=0)
y = pd.read_csv('csv_files/y.csv', header=0)

X = X.iloc[:, :].values
y = y.iloc[:, :].values'''
model_x = XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1009,
                max_depth = 15, alpha = 16, n_estimators = 500)
model_y = XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 16, alpha = 10, n_estimators = 1000)

test = np.array([], dtype=np.int64).reshape(0,2)
pred = np.array([], dtype=np.int64).reshape(0,2)

y_pred = []

X,y = criar_dados(10000, X, y)
X = shuffle(X)
#save_csv(X, 'csv_files/1000.csv')

indices = np.arange(X.shape[0])
rng = np.random.RandomState(123)
permuted_indices = rng.permutation(indices)

train_size, valid_size = int(0.90*X.shape[0]), int(0.10*X.shape[0])
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

print_error(test, pred)
metrica(test, pred)

#%%
''' salvar os frames apenas testando os resultados de um dos arquivos gerados automaticamente '''

data = pd.read_csv("in/csv/test/data_tcc_meio_campo.csv")
label_pes = pd.read_csv("in/csv/label_pes.csv")
label_pes = list(label_pes.columns[:])

data = data.drop(label_pes, axis=1 , errors='ignore')
_INDEX_V = len(data.columns) - 5
_INDEX_X = len(data.columns) - 4
_INDEX_Y = len(data.columns) - 2
_INDEX_ID = len(data.columns)


X = data.iloc[:, 0:_INDEX_V].values
index = data.index.values

_y_pred_x = model_x.predict(X)
_y_pred_y = model_y.predict(X)

y_pred = np.vstack((_y_pred_x, _y_pred_y))
y_pred = y_pred.T

for index, row in data.iterrows():
    print (row[83])
    plt.figure()
    img = row[83]
    #index = row[84]
    
    plt.imshow(io.imread('in/img/' + img))
    plt.title(img)
    plt.plot(y_pred[index][0], y_pred[index][1], 'b*') 
    plt.text(y_pred[index][0]-5, y_pred[index][1]-10, 'predicted ball' )
    
    plt.xlim(0, 1920)
    plt.ylim(1080, 0)
    plt.show() 
    plt.title(img) 
    #plt.pause(0.1)
    #plt.clf()
    save = 'out/videos/data_tcc_meio_campo/'+str(img)
    plt.savefig(save, format='png')
    
    
'''
test = np.vstack((test,y_test))
pred = np.vstack((pred,y_pred))

print_error(test, pred)
metrica(test, pred)
'''


#%% savar in np 

save =  np.concatenate((test, pred), axis=1)

with open('np_files/result_pixel_21_08_2.npy', 'wb') as f:
    np.save(f, test)
    np.save(f, pred)

    