import mat73
import torch

import numpy as np
import matplotlib.pyplot as plt
data = mat73.loadmat('indy_20160915_01.mat')
spike=np.array(data['spikes'])
l_time=data['t']
cursor=data['cursor_pos']
target=data['target_pos']
event={}
valid_directions=['u','r','l','d']
def search(x,time_list):
    l,r=0,len(time_list)-1
    mid=int((l+r)/2)
    while l!=mid:
        if time_list[mid]<=x:
            l=mid           
        else:
            r=mid
        mid=int((l+r)/2)
    return mid

def com_dir(s,d):
    dis_x=d[0]-s[0]
    dis_y=d[1]-s[1]
    dir=''
    if dis_x>0 and dis_y>0:
        if dis_x>dis_y:
            dir='r'
        else:
            dir='u'
    if dis_x>0 and dis_y<0:
        if dis_x>-dis_y:
            dir='r'
        else:
            dir='d'
    if dis_x<0 and dis_y>0:
        if -dis_x>dis_y:
            dir='l'
        else:
            dir='u'
    if dis_x<0 and dis_y<0:
        if dis_x>dis_y:
            dir='d'
        else:
            dir='l'
    return dir

def process():
    all_data=[]
    num_channel=0
    for channel in spike:
        c=[]
        num=0
        num_cell=0
        for cells in channel:
            if num==0:
                num+=1
                continue
            cell={}
            try:
                for j in cells:
                    index=search(j,l_time)
                    dir=com_dir(cursor[index],target[index])
                    if dir not in cell.keys():
                        cell[dir]=[]
                    cell[dir].append(j)
                for v_dir in valid_directions:
                    if not v_dir in cell.keys():
                        cell[v_dir]=[]
                c.append(cell)
            except TypeError:
                continue
        num_cell+=1
        all_data.append(c)
        num_channel+=1
    return all_data

def draw(channel_index,cell_index,data):
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus']=False
    colors1 = ['C{}'.format(i) for i in range(4)]
    plt.eventplot(data,colors=colors1)
    label=['上','右','下','左']
    plt.legend(label)
    plt.savefig('dir_raster/channel_'+str(channel_index)+'_cell_'+str(cell_index)+'.png')
    plt.clf()



for i in range(len(l_time)):
    if not str(target[i]) in event.keys():
        dir=com_dir(cursor[i],target[i])
        dct={}
        dct['ltime']= 0 if i==0 else l_time[i-1]
        dct['rtime']=l_time[i]+200
        dct['direction']=dir
        dct['cell']=[]
        event[str(target[i])]=dct
print(event)
# all_data=process()
# for channel_index in range(len(all_data)):
#     for cell_index in range(len(all_data[channel_index])):
#         data=[]
#         data_cell=all_data[channel_index][cell_index]
#         data.append(data_cell['u'])
#         data.append(data_cell['r'])
#         data.append(data_cell['d'])
#         data.append(data_cell['l'])
#         draw(channel_index,cell_index,data)
