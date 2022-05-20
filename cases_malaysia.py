# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:53:46 2022

@author: End User
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle


from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import mean_absolute_error

#PATHS
PATH_LOGS=os.path.join(os.getcwd(),'accessment4','cases_malaysia_logs')
MODEL_SAVE_PATH=os.path.join(os.getcwd(),'saved_models','model.h5')
SCALER_SAVE_PATH1 = os.path.join(os.getcwd(), "saved_models", "X_scaler.pkl")



#%%pointing to all DATASETS
DATASET_TRAIN_PATH=os.path.join(os.getcwd(),"dataset","cases_malaysia_train.csv")
DATASET_TEST_PATH=os.path.join(os.getcwd(),"dataset","cases_malaysia_test.csv")


#step1)data loading
X_train=pd.read_csv(DATASET_TRAIN_PATH)
X_test=pd.read_csv(DATASET_TEST_PATH)



#step2)data inspection

X_train.info()
X_train.describe().T


'''get vlaues from datset'''
x_train=X_train['cases_new'].values

'''convert the dataset from numpy array to dataframe for easy execution'''
x_train=pd.DataFrame(x_train)

x_train.columns=['cases_new']

''' change non numerical values into NaN values and then convert to 0'''
x_train['cases_new']=pd.to_numeric(x_train['cases_new'],errors='coerce')
x_train['cases_new']=x_train['cases_new'].fillna(0)



x_test=X_test['cases_new'].values




#step3)data visualization
plt.figure()
plt.plot(x_train)
plt.show()

plt.figure()
plt.plot(x_test)
plt.show()

#step4)data cleaning
#step5)feature selection
#step6)data preprocessing
#x_train = np.asfarray(x_train)




#x_train = np.array(x_train).astype('float64')  
'''convert all string values into int/float using MinMaxScaler'''

mms=MinMaxScaler()
X_train_scaled=mms.fit_transform(x_train)
X_test_scaled=mms.fit_transform(np.expand_dims(x_test,-1))
pickle.dump(mms,open('X_scaler.pkl','wb'))

x_train=[]
y_train=[]

x_test=[]
y_test=[]

#training dataset with window size 60 days
'''values of dataset and window size =30days'''
for i in range(30,680):
    x_train.append(X_train_scaled[i-30:i,0])
    y_train.append(X_train_scaled[i,0])



#testing dataset with window size 60 days


dataset_tot=np.concatenate((X_train_scaled,X_test_scaled),axis=0)
data=dataset_tot[:-50:]

x_test=[]
y_test=[]
for i in range(30,50):
    x_test.append(data[i-30:i,0])
    y_test.append(data[i,0])

x_train=np.array(x_train)
x_train=np.expand_dims(x_train,axis=-1)

y_train=np.array(y_train)



x_test=np.array(x_test)
x_test=np.expand_dims(x_test,axis=-1)
y_test=np.array(y_test)


#%% model creation

model=Sequential()
model.add(LSTM(32,activation='tanh',
               return_sequences=True,
               input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1,activation='relu'))
model.summary()

model.compile(optimizer='adam',loss='mse',metrics='mse')


log_files=os.path.join('cases_malaysia_logs')
tensorboard_callback=TensorBoard(log_dir=log_files,histogram_freq=1)





hist=model.fit(x_train,y_train,epochs=5,callbacks=[tensorboard_callback])
print(hist.history.keys())

plt.figure()
plt.plot(hist.history['loss'])
plt.xlabel('epochs')
plt.ylabel('training loss')
plt.show()



#%%model deployment
#prediction
predicted=[]
for i in x_test:
    predicted.append(model.predict(np.expand_dims(i,axis=0)))


actual=[]
for i in x_train:
    actual.append(model.predict(np.expand_dims(i,axis=0)))

plt.figure()
plt.plot(np.array(predicted).reshape(20,1),color='r')
plt.plot(y_test,color='b')
plt.legend(['predicted','actual'])
plt.show()

inversed_y_true=mms.inverse_transform(np.expand_dims(y_test,axis=-1))
inversed_y_predi=mms.inverse_transform(np.array(predicted).reshape(20,1))

#%%performance evaluation

y_true=y_test
y_predicted=np.array(predicted).reshape(20,1)
y_actual=np.array(actual).reshape(650,1)

print('mean absolute error')
print(mean_absolute_error(y_test, y_predicted)/sum(abs(y_actual))*100)



#%%model saving

model.save(MODEL_SAVE_PATH)