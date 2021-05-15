import cv2
import numpy as np
np.set_printoptions(suppress=True) #para print mais bonito
from skimage import transform
from utils.utils import save_np

def persp2(file, path, plot=False, option=1):
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