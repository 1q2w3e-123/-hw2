import mat73
import torch

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
vel_time=''
vel=''
with open('vel.txt','r') as f:
    vel=f.read()
vel=eval(vel)
with open('vel_time.txt','r') as f:
    vel_time=f.read()
vel_time=eval(vel_time)

def get_v(speed):
    return math.sqrt(speed[0]**2+speed[1]**2)

def search_target(index):
    t=target[index]
    while index>0:
        if target[index][0]!=t[0] or target[index][1]!=t[1]:
            break
        index-=1
    return index

def search(x,time_list):
    l,r=0,len(time_list)-1
    mid=int((l+r)/2)
    while l!=mid:
        if time_list[mid]<x:
            l=mid           
        else:
            r=mid
        mid=int((l+r)/2)
    t=target[mid]
    return mid

def process():
    all_speed={}
    num_channel=0
    for channel in spike:
        num=0
        num_cell=0
        for cells in channel:
            if num==0:
                num+=1
                continue
            t_cell=[]
            cell_data={}
            cell_data['time']=[]
            cell_data['speed']=[]
            try:
                if len(cells)<500:
                    continue
                now_t,last_t=target[0],target[0]
                for j in cells:
                    index=search(j,vel_time)
                    start_index=search_target(index)
                    if j<=l_time[start_index]+0.5 and j>=l_time[start_index]:
                        speed=get_v(vel[start_index])
                        t_cell.append(j-l_time[start_index])
                    now_t=target[index]
                    if now_t[0]!=last_t[0] or now_t[1]!=last_t[1]:
                        cell_data['time'].append(t_cell)
                        cell_data['speed'].append(speed)
                        t_cell=[]
                    last_t=now_t
            except TypeError:
                continue
            all_speed["channel"+str(num_channel)+"_cell"+str(num_cell)]=cell_data
            num_cell+=1
        num_channel+=1
    return all_speed

def func(x,a,b):
    return a*x+b

def draw(name,dir,data):
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus']=False
    # colors1 = ['C{}'.format(i) for i in range(4)]
    plt.eventplot(data)
    # plt.legend(label)
    plt.savefig('speed/raster_'+name+'_'+str(dir)+'.png')
    plt.clf()

def draw_psth(name,dir,data):
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus']=False
    # colors1 = ['C{}'.format(i) for i in range(4)]
    y=[]
    for i in data:
        y.extend(i)
        y.extend(i)
    plt.hist(y, bins=20)
    plt.ylabel('firing rate per second')
    plt.savefig('speed/psth_'+name+'_'+str(dir)+'.png')
    plt.clf()

def draw_curve(x1,y1,name,popt):
    x2=np.linspace(0,17.2,100)
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus']=False
    y2=[]

    for i in range(100):
        y2.append(func(x2[i],popt[0],popt[1]))
    plt.scatter(x1, y1, marker='x',lw=1,label='原始数据')
    plt.plot(x2,y2,c='r',label='拟合曲线')
    plt.xticks(x1)
    plt.legend() # 显示label
    plt.savefig('speed/tuning_curve_'+name+'.png')
    plt.clf()

# all_speed=process()
# with open('all_speed.txt','w') as f:
#     f.write(str(all_speed))
s=''
with open('all_speed.txt','r') as f:
    s=f.read()
all_speed_data=eval(s)
# speed
curve=[]
from sklearn.metrics import r2_score
num=0

for name in all_speed_data.keys():
    cell=all_speed_data[name]
    cell_s=cell['speed']
    l=min(cell_s)
    r=20
    dt=2
    edges=np.arange(l,r,dt)
    psth=np.histogram(cell_s,edges)
    all_speed=psth[1]
    cos_y=psth[0]
    cell_data={}
    cos_x=[]
    for i in range(len(all_speed)-1):
        cell_data[str(all_speed[i])+'~'+str(all_speed[i+1])]=[]
        cos_x.append((all_speed[i]+all_speed[i+1])/2)
    for i in range(len(cell_s)):
        index=search(cell_s[i],all_speed)
        cell_data[str(all_speed[index])+'~'+str(all_speed[index+1])].append(cell['time'][i])
    for key in  cell_data.keys():
        draw(name,key,cell_data[key])
        draw_psth(name,key,cell_data[key])
    if num==0:
        popt, pcov = curve_fit(func, cos_x, cos_y)
        curve=popt.tolist()
        draw_curve(cos_x,cos_y,name,popt)
    # else:
    y_pred=[]
    for i in cos_x:
        y_pred.append(func(i,curve[0],curve[1]))
    # y_pred=func(cos_x,curve[0],curve[1])
    r2=r2_score(cos_y,y_pred)
    with open('speed_r2_score.txt','a') as f:
        f.write(name+': '+str(r2)+'\n')
    num+=1




    # break
    # cell=np.array(cell)
    # print(cell)
    # start=min(cell)
    # end=max(cell)
    # print(start,end)
    # dt=5
    # edges=np.arange(start,end,dt)
    # psth=np.histogram(cell,edges)[0]
    # # draw()
    # print(psth)
    # break