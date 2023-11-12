#Import standard packages
import numpy as np
import matplotlib.pyplot as plt
import mat73
from scipy import io
from scipy import stats
import pickle

from Neural_Decoding.metrics import get_R2
from Neural_Decoding.metrics import get_rho
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
#Import decoder functions
from Neural_Decoding.decoders import KalmanFilterDecoder
from Neural_Decoding.preprocessing_funcs import bin_output

with open('data.pickle','rb') as f:
#     neural_data,vels_binned=pickle.load(f,encoding='latin1') #If using python 3
    neural_data,vels_binned,pos_binned=pickle.load(f)
X_kf=neural_data

# pos_binned=np.zeros(vels_binned.shape) #Initialize 
# pos_binned[0,:]=0 #Assume starting position is at [0,0]
# #Loop through time bins and determine positions based on the velocities
# for i in range(pos_binned.shape[0]-1): 
#     pos_binned[i+1,0]=pos_binned[i,0]+vels_binned[i,0]*.1 #Note that .05 is the length of the time bin
#     pos_binned[i+1,1]=pos_binned[i,1]+vels_binned[i,1]*.1

#We will now determine acceleration    
temp=np.diff(vels_binned,axis=0) #The acceleration is the difference in velocities across time bins 
acc_binned=np.concatenate((temp,temp[-1:,:]),axis=0) #Assume acceleration at last time point is same as 2nd to last

#The final output covariates include position, velocity, and acceleration
y_kf=np.concatenate((pos_binned,vels_binned,acc_binned),axis=1)
# y_kf=np.concatenate((pos_binned,acc_binned),axis=1)
# y_kf=np.concatenate((pos_binned,vels_binned),axis=1)
# y_kf=np.concatenate((vels_binned,acc_binned),axis=1)
# y_kf=pos_binned
# y_kf=vels_binned
# y_kf=acc_binned



training_range=[0, 0.7]
testing_range=[0.7, 0.85]
valid_range=[0.85,1]

num_examples_kf=X_kf.shape[0]
        
#Note that each range has a buffer of 1 bin at the beginning and end
#This makes it so that the different sets don't include overlapping data
training_set=np.arange(np.int(np.round(training_range[0]*num_examples_kf))+1,np.int(np.round(training_range[1]*num_examples_kf))-1)
testing_set=np.arange(np.int(np.round(testing_range[0]*num_examples_kf))+1,np.int(np.round(testing_range[1]*num_examples_kf))-1)
valid_set=np.arange(np.int(np.round(valid_range[0]*num_examples_kf))+1,np.int(np.round(valid_range[1]*num_examples_kf))-1)

#Get training data
X_kf_train=X_kf[training_set,:]
y_kf_train=y_kf[training_set,:]

#Get testing data
X_kf_test=X_kf[testing_set,:]
y_kf_test=y_kf[testing_set,:]

#Get validation data
X_kf_valid=X_kf[valid_set,:]
y_kf_valid=y_kf[valid_set,:]

X_kf_train_mean=np.nanmean(X_kf_train,axis=0)
X_kf_train_std=np.nanstd(X_kf_train,axis=0)
X_kf_train=(X_kf_train-X_kf_train_mean)/X_kf_train_std
X_kf_test=(X_kf_test-X_kf_train_mean)/X_kf_train_std
X_kf_valid=(X_kf_valid-X_kf_train_mean)/X_kf_train_std

#Zero-center outputs
y_kf_train_mean=np.mean(y_kf_train,axis=0)
y_kf_train=y_kf_train-y_kf_train_mean
y_kf_test=y_kf_test-y_kf_train_mean
y_kf_valid=y_kf_valid-y_kf_train_mean

kal_C=3
model_kf=KalmanFilterDecoder(C=kal_C) #There is one optional parameter that is set to the default in this example (see ReadMe)

#Fit model
model_kf.fit(X_kf_train,y_kf_train)

#Get predictions
y_valid_predicted_kf=model_kf.predict(X_kf_valid,y_kf_valid)


# R2_kf=get_R2(y_kf_valid,y_valid_predicted_kf)
# print('R2:',R2_kf[2:4]) #I'm just printing the R^2's of the 3rd and 4th entries that correspond to the velocities
# rho_kf=get_rho(y_kf_valid,y_valid_predicted_kf)
# print('rho2:',rho_kf[2:4]**2)

r2_kal=r2_score(y_kf_valid,y_valid_predicted_kf)
mse_kal=mean_squared_error(y_kf_valid,y_valid_predicted_kf)
print(r2_kal,mse_kal)
with open('results_init.txt','a') as f:
    f.write('C='+str(kal_C)+':\n')
    f.write(str(r2_kal)+'\n\n')
assert(1==0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_kf_train,y_kf_train)
y_pred=lr.predict(X_kf_valid)
r2_linear=r2_score(y_kf_valid,y_pred)
mse_linear=mean_squared_error(y_kf_valid,y_pred)
print(r2_linear,mse_linear)

from keras.layers.recurrent import LSTM
from keras.models import Sequential
model = Sequential()
model.add(LSTM(6, input_shape=(X_kf_train.shape[0],X_kf_train.shape[1]), return_sequences=True))
model.compile( loss="mse", optimizer="rmsprop" )
X_kf_train=np.expand_dims(X_kf_train, axis=0)
y_kf_train=np.expand_dims(y_kf_train, axis=0)
model.fit(X_kf_train,y_kf_train)
print(X_kf_train.shape,y_kf_train.shape)
X_kf_valid=np.expand_dims(X_kf_valid, axis=0)
y_pred=model.predict(X_kf_valid)
print(y_kf_valid.shape,y_pred.shape)
r2_lstm=r2_score(y_kf_valid,y_pred[0])
mse_lstm=mean_squared_error(y_kf_valid,y_pred[0])
print(r2_lstm,mse_lstm)

with open('results.txt','a') as f:
    f.write("pos acc vel:\n")
    f.write('R2_score:\n')
    f.write('kal:'+str(r2_kal)+'\n')
    f.write('linear:'+str(r2_linear)+'\n')
    f.write('lstm:'+str(r2_lstm)+'\n')
    f.write('MSE:\n')
    f.write('kal:'+str(mse_kal)+'\n')
    f.write('linear:'+str(mse_linear)+'\n')
    f.write('lstm:'+str(mse_lstm)+'\n\n')
