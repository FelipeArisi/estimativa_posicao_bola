import numpy as np
import cv2
from matplotlib import pyplot as plt
from utils.utils import save_np

# Realiza a correção dos pixes devido a lente da camera

def correcao(file, path, plot=False):
    DIM = (1920,1080)
    K=np.array([[809.155038, 0.0, 960.0], [0.0, 494.790948, 540.0], [0.0, 0.0, 1.0]])
    #D=np.array([[1.0000000000000000e+00],[-1.0927006057935250e-07],[-1.2372445556263467e-13],[0]])
    D=np.array([1.0000000000000000e-01, -1.0927006057935250e-07, -1.2372445556263467e-13, 0])
    
    with open(path+file, 'rb') as f:
        test = np.load(f)
        pred = np.load(f)
    
    img_path = '../in/img/00050.png'
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    
    newK = np.array([[809.155038, 0.0, DIM[0]], [0.0, 494.790948, DIM[1]], [0.0, 0.0, 1.0]])
    
    #map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, None, K, (w,h), 5)
    #undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    undistorted_img = cv2.fisheye.undistortImage(img, K, D, np.eye(3), newK, (DIM[0]*2,DIM[1]*2))
    undistorted_img = cv2.resize(undistorted_img, DIM)
    
    velho = img.copy()[:,:,::-1] #converte BGR para RGB
    novo = undistorted_img.copy()[:,:,::-1]
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(velho)
    pts = np.array([[ [900.0,500.0], [1100,600], [200.0, 300.0], [1800,700]]])
    #pts = np.array([[ [531,347.0],[381,420.0], [683,381.0], [927,339.0], [880,358.0], [851,691.0], [903,476.0], [1289,511.0], [1490,689.0], [1480,403.0], [1336,423.0], [1344,329.0], [511,518.0], [536,692.0], [1086,342.0], [1176,692.0] ]])
    #pts = np.array([test])
    for x,y in pts[0]:
        plt.plot(x,y,'ro')
    
    plt.subplot(122)
    plt.imshow(novo)
    aux = cv2.fisheye.undistortPoints(pts, K, D, P=newK)
    aux = aux/2
    print(aux)
    if(plot):
        for x,y in aux[0]:
            plt.plot(x,y,'ro')

    pred = (cv2.fisheye.undistortPoints( np.array([pred]), K, D, P=newK))/2
    test = (cv2.fisheye.undistortPoints( np.array([test]), K, D, P=newK))/2
    
    
    info_file = save_np(test, pred)
    
    return info_file

'''with open('processing/np/result_pixel_correcao_22_08_2.npy', 'wb') as f:
    np.save(f, test)
    np.save(f, pred)'''
    