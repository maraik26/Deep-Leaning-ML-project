
# coding: utf-8

#get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import scipy.special as sps
import warnings
import keras.backend as K
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

data=pd.read_csv("preprocessed_datasets.csv")

draft_year=data["DraftYear"]
sum_7yr_gp_ori=data["sum_7yr_GP"]
data.drop(["id", "PlayerName", "sum_7yr_TOI", "Overall", "GP_greater_than_0","Country"], axis=1, inplace=True)

def standardize(df):
    res = df.copy()
    for i in res.columns:
        mean=np.mean(res[i])
        sd=np.std(res[i])
        if(sd == 0):
             res.drop(i, axis=1, inplace=True)
        else:
             res[i]=(res[i]-mean)/sd
    return res

train_set=data[data['DraftYear'].isin([2004,2005,2006])]
test_set=data[data['DraftYear'].isin([2007])]

sum_7yr_gp_train=train_set["sum_7yr_GP"]
sum_7yr_gp_test=test_set["sum_7yr_GP"]

sum_7yr_gp_train.hist()

train_set.drop(['DraftYear',"sum_7yr_GP"], axis=1, inplace=True)
test_set.drop(['DraftYear',"sum_7yr_GP"], axis=1, inplace=True)

train_set = pd.get_dummies(train_set, columns= ['country_group','Position'])
test_set = pd.get_dummies(test_set, columns= ['country_group','Position'])
data.drop(['DraftYear',"sum_7yr_GP"], axis=1, inplace=True)

scaler = StandardScaler()
scaler.fit(train_set)
scaler_y = StandardScaler()
scaler_y.fit(sum_7yr_gp_train.reshape(-1, 1))
#print(scaler.mean_)

train_set_final = scaler.transform(train_set)
test_set_final = scaler.transform(test_set)

sum_7yr_gp_train = sum_7yr_gp_train.astype('float32')

#KERAS model built

model = Sequential()

def rsquared(y_true, y_pred):
    s_res = K.sum(K.square(y_true - y_pred))
    s_tot = K.sum(K.square(y_true - K.mean(y_true)))
    r2 = 1 - s_res/(s_tot + K.epsilon())
    return(r2)

#model.add(Dense(units=1, activation='linear', input_shape=(22,)))

model.add(Dense(units=1, activation='linear', input_shape=(22,)))
#model.add(Dense(units=1, activation='linear'))

sgd = keras.optimizers.SGD(lr=1, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam()
rmsprop = keras.optimizers.RMSprop(lr=0.001)
model.compile(loss='mean_squared_error',
              optimizer=adam,
              metrics=[rsquared])

# model.add(Dropout(0.3))
# model.add(Dense(units=1, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(units=1, activation='linear', input_shape=(22,)))
# model.add(Dropout(0.8))

model.summary()

history = model.fit(np.array(train_set_final), sum_7yr_gp_train, epochs=110, 
          batch_size=32, shuffle=True, validation_data=(np.array(test_set_final), sum_7yr_gp_test))

hist_df = pd.DataFrame(history.history)

fig = plt.figure(figsize=(14,6))
plt.style.use('bmh')
params_dict = dict(linestyle='solid', linewidth=0.25, marker='o', markersize=6)

plt.subplot(121)
plt.plot(hist_df.loss, label='Training loss', **params_dict)
plt.plot(hist_df.val_loss, label='Validation loss', **params_dict)
plt.title('Loss for ' + str(len(history.epoch)) + ' epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(122)
plt.plot(hist_df.rsquared, label='Training accuracy', **params_dict)
plt.plot(hist_df.val_rsquared, label='Validation accuracy', **params_dict)
plt.title('Accuracy for ' + str(len(history.epoch)) + ' epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.hist(model.predict(np.array(train_set_final)))
plt.hist(sum_7yr_gp_train, bins=50)
plt.hist(scaler_y.inverse_transform(model.predict(np.array(train_set_final))))
plt.hist(scaler_y.inverse_transform(sum_7yr_gp_train), bins=50)

