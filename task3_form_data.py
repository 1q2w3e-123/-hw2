import mat73
import torch

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import pylab as mpl
from Neural_Decoding.preprocessing_funcs import bin_spikes
from Neural_Decoding.preprocessing_funcs import bin_output

data = mat73.loadmat('indy_20160915_01.mat')
spike=np.array(data['spikes'])
l_time=data['t']
cursor=data['cursor_pos']
target=data['target_pos']
def process():
    spike_time=[]
    for channel in spike:
        flag=0
        for cells in channel:
            if flag==0:
                flag=1
                continue
            try:
                if len(cells)<500:
                    continue
                spike_time.append(cells.tolist())
            except:
                continue
    return np.array(spike_time)


vel=''
vel_time=''
with open('vel.txt','r') as f:
    vel=f.read()
with open('vel_time.txt','r') as f:
    vel_time=f.read()
vel=np.array(eval(vel))
vel_time=np.array(eval(vel_time))
spike_time=process()
dt=.1
t_start=vel_time[0]
t_end=vel_time[-1]
downsample_factor=1

neural_data=bin_spikes(spike_time,dt,t_start,t_end)
vels_binned=bin_output(vel,vel_time,dt,t_start,t_end,downsample_factor)
pos_binned=bin_output(cursor,vel_time,dt,t_start,t_end,downsample_factor)

import pickle

data_folder='' #FOLDER YOU WANT TO SAVE THE DATA TO

with open(data_folder+'data.pickle','wb') as f:
    pickle.dump([neural_data,vels_binned,pos_binned],f)
