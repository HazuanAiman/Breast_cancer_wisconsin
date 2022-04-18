# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:47:29 2022

@author: User
"""
#https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
import pandas as pd
import tensorflow as tf
import datetime as dt

data = pd.read_csv('X:/Users/User/Tensorflow Deep Learning/csv/breast cancer wisconsin.csv', header=0)

#%%
#Data preparation
bcw = data.drop(["id", "Unnamed: 32"], axis=1)
feature = bcw.iloc[:, 1:]
label = bcw.iloc[:, 0]

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
feature = ss.fit_transform(feature)

label = pd.get_dummies(label)
# ohe= OneHotEncoder(categorical_features = [0])    
# x= ohe.fit_transform(bcw).toarray()

#%%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=12345)

#%%
input = x_train.shape[-1]
output = y_train.shape[-1]

model = tf.keras.models.Sequential()
#model.add(normalizer)
model.add(tf.keras.Input(shape=input))
model.add(tf.keras.layers.Dense(64, activation='sigmoid'))
model.add(tf.keras.layers.Dense(32, activation='sigmoid'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(output, activation='sigmoid'))

model.summary()

#%%
log_path = r"X:\Users\User\Tensorflow Deep Learning\Tensorboard\breastcancerwiconsin" + dt.datetime.now().strftime("%H%M%S")
es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1)
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=100, callbacks=[es_callback, tb_callback])
#tf.keras.utils.plot_model(model, show_shapes=True)