# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:31:29 2020

@author: felipe
"""
import json
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from data import return_text_tracked
import random

def axis(keypoints):
    points = {'x': [], 'y':[], 'score':[]}
    points['x'] = [keypoints[i] for i in range(0, 51, 3)] 
    points['y'] = [keypoints[i] for i in range(1, 51, 3)] 
    points['score'] = [keypoints[i] for i in range(2, 51, 3)] 
    return points

def correct_vector(array):
    while(len(array)<10):
        array.append('')
    return array

with open('test/alphapose-tracked_2.json', 'r') as json_file:
    tracked = json.load(json_file)
    

    
#with open('test/alphapose-results-forvis.json', 'r') as json_file:
#    forvis = json.load(json_file)
    
#with open('test/alphapose-results.json', 'r') as json_file:
#    results = json.load(json_file)

#print(return_text_tracked('04073.png',[243,219,271,256,177,292,211,'',276,163], 1,362.16,511.5))

not_players = [2,10,74,41,21,181,201,191,192,289,320,298,426,433,413,423]
i=0
players_frame = []
array_text = []
for img in sorted(tracked)[976:1004]: # percorrer os frames :265
    #plt.figure()
    players_frame.append([])
    plt.imshow(io.imread('img/' + img))
    for person in tracked[img]: # percorrer as pessoas do frame
        keypoints = person['keypoints'] 
        points = axis(keypoints)   
        if(person['scores'] > 1.61 and keypoints[49] < 750.0 and keypoints[49] > 180.0 and person['idx'] not in not_players):
            plt.plot(points['x'], points['y'], 'ko-') 
            plt.text(points['x'][0]-5, points['y'][0]-10, str(person['idx']) +' - ' +str( round(person['scores'],2) ))
            players_frame[i].append(person['idx'])
    random.shuffle(correct_vector(players_frame[i]))    
    array_text.append(return_text_tracked(img,players_frame[i], 0,0,0))     
    i=i+1
    plt.xlim(0, 1920)
    plt.ylim(1080, 0)
    plt.show()
    plt.pause(0.01)
    plt.clf()
    plt.title(img)
    
    
np.savetxt('data_interactive.csv', [p for p in array_text], delimiter=',', fmt='%s')
