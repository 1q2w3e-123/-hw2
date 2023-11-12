import mat73
import torch

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import pylab as mpl
from sklearn.metrics import r2_score
def func(x,a,c,d):
    return a*np.cos(x+c)+d

def com_mse(y, t):
    y=np.array(y)
    t=np.array(t)
    return 0.5 * np.sum((y - t)**2)

cos_x=[0,math.pi/2,math.pi,math.pi*1.5]
fire_rate=''
curve=''
with open('fire_rate.txt','r') as f1:
    fire_rate=f1.read()
with open('curve.txt','r') as f2:
    curve=f2.read()

fire_rate=eval(fire_rate)
curve=eval(curve)

for name in curve.keys():
    all_mse={}
    for cell in fire_rate.keys():
        y1=fire_rate[cell]
        y2=[]
        for i in cos_x:
            y2.append(func(i,curve[name][0],curve[name][1],curve[name][2]))
        mse=r2_score(y1,y2)
        all_mse[cell]=mse
    with open('R2/'+name+'.txt','w') as f:
        f.write(str(all_mse))


