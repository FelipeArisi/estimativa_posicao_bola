import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#df = pd.read_csv('resultados_teste.csv', header=1)

#m = df.to_numpy('float')/10

with open('np_files/result_metros_22_08_2.npy', 'rb') as f:
    m = np.load(f)/10

cols = (m[:,0] - m[:,2])**2
lins = (m[:,1] - m[:,3])**2

dis = np.sqrt( cols + lins)

hist, bins = np.histogram(dis, 50)
plt.bar(np.arange(0, len(hist)), hist)
plt.xticks(np.arange(0, len(hist)), ['%.1f'%x for x in bins], rotation=-45)
plt.title('Media = %.2f ; Mediana = %.2f' %(dis.mean(), np.median(dis)))