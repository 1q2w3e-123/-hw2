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
vel=[]
vel_time=[]
acc=[]
event={}
cos_x=[]
sum_dir=20
for i in range(sum_dir):
    cos_x.append(i/sum_dir*2*math.pi)
valid_directions=['r','u','l','d']
cos_x=[0,math.pi/2,math.pi,math.pi*1.5]

def compu_vel():
    for i in range(1,len(l_time)-1):
        vel_x=(cursor[i+1][0]-cursor[i-1][0])/(l_time[i+1]-l_time[i-1])
        vel_y=(cursor[i+1][1]-cursor[i-1][1])/(l_time[i+1]-l_time[i-1])
        v=[vel_x,vel_y]
        vel.append(v)
        vel_time.append(l_time[i])

def compu_acc():
    for i in range(0,len(vel)):
        if i==0:
            acc_x=(vel[i][0]-0)/(vel_time[i]-l_time[i])
            acc_y=(vel[i][1]-0)/(vel_time[i]-l_time[i])
            continue
        if i==len(vel)-1:
            acc_x=((0-vel[i][0])/(l_time[i+1]-vel_time[i]))
            acc_x=((0-vel[i][1])/(l_time[i+1]-vel_time[i]))
            continue
        acc_x=(vel[i+1][0]-vel[i-1][0])/(vel[i+1]-vel[i-1])
        acc_y=(vel[i+1][1]-vel[i-1][1])/(vel[i+1]-vel[i-1])
        a=[acc_x,acc_y]
        acc.append(a)

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

# 

def process():
    all_data={}
    num_channel=0
    for channel in spike:
        flag=0
        num_cell=0
        for cells in channel:
            if flag==0:
                flag=1
                continue
            cell={}
            try:
                if len(cells)==0:
                    continue
                for d in valid_directions:
                    cell[d]=[]
                    cell[d+'1']=[]
                now_t,last_t=target[0],target[0]
                l=[]
                l1=[]
                # print(len(cells))
                if len(cells)<500:
                    continue
                for signal in cells:
                    index=search(signal,l_time)
                    start_index=search_target(index)
                    if signal<=l_time[start_index]+0.5 and signal>=l_time[start_index]:
                        l.append(signal-l_time[start_index])
                        l1.append(signal)
                    now_t=target[index]
                    if now_t[0]!=last_t[0] or now_t[1]!=last_t[1]:
                        cell[event[str(last_t)]['direction']].append(l)
                        cell[event[str(last_t)]['direction']+'1'].append(l1)
                        l=[]
                        l1=[]
                    last_t=now_t
                # print(cell['r'])
                # assert(1==0)
                all_data['channel'+str(num_channel)+'_cell'+str(num_cell)]=cell
                num_cell+=1
            except TypeError:
                continue
        num_channel+=1
    return all_data

def draw(name,dir,data):
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus']=False
    # colors1 = ['C{}'.format(i) for i in range(4)]
    plt.eventplot(data)
    # plt.legend(label)
    plt.savefig('raster/'+name+"_"+dir+'.png')
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
    plt.savefig('psth/'+name+"_"+dir+'.png')
    plt.clf()

def draw_curve(x1,y1,name,popt):
    x2=np.linspace(0,2*math.pi,100)
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus']=False
    y2=[]
    for i in range(100):
        y2.append(func(x2[i],popt[0],popt[1],popt[2]))
    plt.scatter(x1, y1, marker='x',lw=1,label='原始数据')
    plt.plot(x2,y2,c='r',label='拟合曲线')
    plt.xticks(x1)
    plt.legend() # 显示label
    plt.savefig('tuning_curve/'+name+'.png')
    plt.clf()

l=[]
t=target[0]
last_i=0
for i in range(len(l_time)):
    if target[i][0]!=target[last_i][0] or target[i][1]!=target[last_i][1]:
        l.append(l_time[i]-l_time[last_i])
        last_i=i
l.sort(reverse=True)
print(l)

def func(x,a,c,d):
    return a*np.cos(x+c)+d

for i in range(len(l_time)):
    if not str(target[i]) in event.keys():
        dir=com_dir(cursor[i],target[i])
        dct={}
        dct['ltime']= 0 if i==0 else l_time[i-1]
        dct['rtime']=l_time[i]+200
        dct['direction']=dir
        dct['cell']=[]
        event[str(target[i])]=dct
all_data=process()
fire_rate={}
curve={}

for i in all_data.keys():
    if len(i)>1:
        continue
    fire_rate[i]=[]
    curve[i]=[]
    for j in  valid_directions:
        fire_rate[i].append(len(all_data[i][j])*2)
    a0 = (max(fire_rate[i])-min(fire_rate[i]))/2
    p0 = [a0,0,0]
    popt, pcov = curve_fit(func, cos_x, fire_rate[i])
    curve[i]=popt.tolist()
    draw_curve(cos_x,fire_rate[i],i,popt)

compu_vel()
compu_acc()
with open('all_data.txt','w') as f:
    f.write(str(all_data))
with open('fire_rate.txt','w') as f:
    f.write(str(fire_rate))
with open('curve.txt','w') as f:
    f.write(str(curve))
with open('vel.txt','w') as f:
    f.write(str(vel))
with open('vel_time.txt','w') as f:
    f.write(str(vel_time)) 
with open('acc.txt','w') as f:
    f.write(str(acc)) 
for i in all_data.keys():
    for j in valid_directions:
        draw(i,j,all_data[i][j])
        draw_psth(i,j,all_data[i][j])
