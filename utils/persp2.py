import cv2
import numpy as np
np.set_printoptions(suppress=True) #para print mais bonito
from skimage import transform
from utils.utils import save_np
from utils.correcao import correcao_img

import matplotlib.pyplot as plt
from skimage import io
import random
# PI sem correção
#PI = np.array([[310,131,491,905,818,760,848,1539,1736,1732,1581,1551,235,298,1180,1340],
#               [247,368,271,187,216,817,416,491,755,337,351,217,505,778,197,808]])

# PI com correçaõ
PI = np.array([[531,381,683,927,880,851,903,1289,1490,1480,1336,1344,511,536,1086,1176],
              [347,420,381,339,358,691,476,511,689,403,423,329,518,692,342,692]])

PQ = np.array([[0,0,101,201,180,201,201,300,301,401,341,401,101,151,276,251],
               [0,84,50,0,21,201,131,150,201,84,100,0,151,201,11,201]])


#PQ = np.array([[0, 0, 0, 0, 61, 101, 101, 101, 126, 126, 151, 181, 181, 201, 201, 201, 201, 401, 401, 401, 401],
 #              [117, 85, 25, 0, 100, 151, 100, 50, 27, 11, 201, 27, 11, 201, 130, 70, 0, 117, 85, 25, 0]])


pt1 = np.column_stack(PI)
pt2 = np.column_stack(PQ)

def persp2_img(img_path):
    im =  correcao_img (img_path)
    M = cv2.findHomography(pt1,pt2)
    test2transf = cv2.perspectiveTransform(im, M[0])
    return test2transf[0]


def persp2_data(data, option = 1, img = None, score = 0):
    if(option == 1):
        M = cv2.findHomography(pt1,pt2)
    else:
        t = transform.ProjectiveTransform()
        t.estimate(pt2, pt1)
    
    lin, col = np.where(np.isnan(data))
    _i=0
    _j=2
    if(img != None):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Conversion to meters image ' + img)

        ax1.axis(xmin=0,xmax=1920)
        ax1.axis(ymin=1080,ymax=0)
        
        ax2.axis(xmin=-5,xmax=45)
        ax2.axis(ymin=25,ymax=-5)

        ax1.imshow( correcao_img (img))
        
        #im_out = cv2.warpPerspective(cv2.imread(img),M[0], (40,20))
        #ax2.imshow(im_out)
    persp = data.copy()
    while(_j <= data.shape[1]):
        if(option == 1):
            persp[:,_i:_j] = cv2.perspectiveTransform( np.array([data[:,_i:_j]]), M[0])/10
        else:
            persp[:,_i:_j] = cv2.perspectiveTransform(np.array([data[:,_i:_j]], dtype=np.float32), t.params)
        if(img != None):
            #print(persp[:,_i:_j][0][0])
            #print(persp[:,_i:_j][0][1])
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)

            ax1.plot(data[:,_i:_j][0][0], data[:,_i:_j][0][1],'o', color=color)
            ax2.plot(persp[:,_i:_j][0][0], persp[:,_i:_j][0][1],'o', color=color)
        _i+=2 + score
        _j+=2 + score
        
    retorno = persp.clip(0)
    retorno[lin,col] = np.nan
    return retorno
    
def persp2(file, path, plot=False, option=1):

    
    with open(path+file, 'rb') as f:
        test = np.load(f)
        pred = np.load(f)
    
    '''with open('np_files/result_pixel_correcao_22_08_2.npy', 'rb') as f:
        test = np.load(f)
        pred = np.load(f)'''
    #t = transform.ProjectiveTransform()
    #t.estimate(pt2, pt1)
    
    #opção 1
    if(option == 1):
        M = cv2.findHomography(pt1,pt2)
        print(M[0])
        
        test2transf = cv2.perspectiveTransform(test, M[0])
        predtransf = cv2.perspectiveTransform(pred, M[0])
    elif(option == 2):
        # na opção 2 arrumar as variaveis para salvar como np
        #opção 2
        t = transform.ProjectiveTransform()
        t.estimate(pt2, pt1)
        print(t.params)
        
        pts2transf = cv2.perspectiveTransform(np.array([test], dtype=np.float32), t.params)
        print(pts2transf[:,0:5, :])
    
    
    
    # Plotar img 
    if(plot):
        import matplotlib.pyplot as plt
        from skimage import io
        
        plt.figure()
        plt.imshow(io.imread('../in/img/04284.png'))
        plt.plot(PI[0], PI[1], 'ko') 
        #plt.xlim(0, 1920)
        #plt.ylim(1080, 0)
        plt.show() 
    
    '''
    plt.figure()
    plt.imshow(io.imread('img_1/03307.png'))
    plt.plot(PI[0], PI[1], 'ko') 
    #plt.xlim(0, 1920)
    #plt.ylim(1080, 0)
    plt.show() 
    
    img = cv2.imread('img/00041.png')
    im_dst = cv2.warpPerspective(img, M[0], (400,200))
    plt.imshow(im_dst)
    
    img = cv2.imread('img/00041.png')
    nova = transform.warp(img, t)
    cropped = nova[0:202,0:402]
    plt.imshow(cropped) 
    
    
    save =  np.concatenate((test2transf[0], predtransf[0]), axis=1)
    info_file = save_np(save, None)
    return info_file'''
    
    save =  np.concatenate((test2transf[0], predtransf[0]), axis=1)
    with open('processing/numpy/result_metros_14_05.npy', 'wb') as f:
        np.save(f, save)
    
    return {'name': 'result_metros_14_05.npy', 'path': 'processing/numpy/'}