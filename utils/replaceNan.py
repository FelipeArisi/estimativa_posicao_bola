# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 17:09:40 2021

@author: felip
"""

import numpy as np

def replaceNan(data, Y):    
    #seleciona jogadores
    jogs = data[:, :-2] #retira 2 ultimas colunas
    
    bola = data[:, -2:] #posicao anterior da bola
    
    novo = np.ones(data.shape) * np.nan
    
    novoY = np.zeros(Y.shape)
    
    #linhas que possuem nan
    com_nan = jogs[np.isnan(jogs).any(axis=1), :]
    #linhas sem nan
    sem_nan = jogs[~np.isnan(jogs).any(axis=1), :]
    divisao = sem_nan.shape[0]
    
    #pega a primeira coordenada de cada jogador
    pe1 = sem_nan[:, 0::4]
    #ordena cada linha da esquerda para direita
    for L in range(sem_nan.shape[0]):
        ordem = np.argsort(pe1[L,:])
        idx = 0
        for pos in ordem:
            novo[L, idx: idx+4] = sem_nan[L, pos*4:(pos*4)+4]
            idx = idx + 4                   
    
    #medias de cada coluna
    medias = np.mean(novo[:divisao, :], axis=0)
    
    lin = divisao
    #agora percorre as linhas com nan
    #e associa cada jogador na coluna mais parecida com a media
    for L in range(com_nan.shape[0]):
        for j in range(0,80,4):
            if ~np.isnan(com_nan[L, j]): #faz isso só pra jogadores não NAN
                dists = np.zeros((20,1))
                jog = com_nan[L, j:j+4]
                for m in range(20):
                    dists[m] = ((jog - medias[m:m+4])**2).sum()
                #menor distancia
                dmin = np.argmin(dists)
                #copia o jogador para a posição q "faz mais sentido"
                novo[lin, dmin:dmin+4] = jog   
        
        lin = lin + 1
            
    #agora preenche todos nan com as medias
    for L in range(divisao, novo.shape[0]):
        for C in range(novo.shape[1]):
            if np.isnan(novo[L,C]):
                novo[L,C] = medias[C]
    
    #copia 2 ultimas colunas (bola) de volta
    #primeiro as linhas q não tinham nan
    novo[~np.isnan(jogs).any(axis=1), -2:] = bola[~np.isnan(jogs).any(axis=1), :]
    #agora as linhas q tinham nan
    novo[np.isnan(jogs).any(axis=1), -2:] = bola[np.isnan(jogs).any(axis=1), :]
    
    #corrige os Y para ficarem na mesma ordem
    novoY = Y.copy()
    novoY[~np.isnan(jogs).any(axis=1), -2:] = Y[~np.isnan(jogs).any(axis=1), :]
    novoY[np.isnan(jogs).any(axis=1), -2:] = Y[np.isnan(jogs).any(axis=1), :]
    
    return novo, novoY  
    