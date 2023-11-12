import mat73
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import pylab as mpl
data = mat73.loadmat('indy_20160915_01.mat')
spike=np.array(data['spikes'])
l_time=data['t']
cursor=data['cursor_pos']
target=data['target_pos']

def draw(data_draw):
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus']=False
    plt.hist(data_draw, bins=20)
    plt.ylabel('num_cell')
    plt.xlabel('r2_score')
    plt.savefig('percentage.png')
    plt.clf()


path='R2'
files=os.listdir(path)
with open(path+'/'+files[0],'r') as f:
    s=f.read()
r2=eval(s)
all_r2=[]
for i in r2.keys():
    if r2[i]<0:
        all_r2.append(0)
        continue
    all_r2.append(r2[i])
# start=0
# end=max(all_r2)
# t=0.1
# edges=np.arange(start,end,t)
# psth=np.histogram(all_r2,edges)
# print(psth,len(all_r2))
# data_draw=[]
# for i in all_r2:
#     data_draw.append(i/len(all_r2))
draw(all_r2)