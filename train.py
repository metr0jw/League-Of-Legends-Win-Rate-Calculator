import numpy as np
import os
from time import time
"""
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import plaidml.keras
plaidml.keras.install_backend()

import keras
"""
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from matplotlib import pyplot as plt


def plot_history(histories, key='binary_crossentropy'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_', ' ').title())
  plt.legend()

  plt.xlim([0, max(history.epoch)])


dataset_location = './dataset/preprocessed/'
x_data = pd.read_csv(dataset_location + 'x_data.csv')
y_data = pd.read_csv(dataset_location + 'y_data.csv')
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)

print('Shape of x_train', x_train.shape)
print('Shape of x_test', x_test.shape)
print('Shape of y_train', y_train.shape)
print('Shape of y_test', y_test.shape)

n_fold = 5
kfold = KFold(n_splits=n_fold, shuffle=True, random_state=0)

accuracy = list()
mean_acc = 0

for i, (train_index, val_index) in enumerate(kfold.split(x_train, y_train.idxmax(1))):
    x_train_kf, x_val_kf = x_train.iloc[train_index, :], x_train.iloc[val_index, :]
    y_train_kf, y_val_kf = y_train.iloc[train_index, :], y_train.iloc[val_index, :]

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(47,)),
        keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                           activation='relu'),
        keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.001),
                           activation='relu'),
        keras.layers.Dense(8, kernel_regularizer=keras.regularizers.l2(0.001),
                           activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(0.001),
        metrics=['accuracy', 'binary_crossentropy']
    )
    tensorboard = TensorBoard(log_dir=r"\logs\{}".format(time()))
    model.fit(x_train_kf,
              y_train_kf,
              epochs=10,
              batch_size=512,
              callbacks=[tensorboard],
              validation_data=(x_val_kf, y_val_kf),
              verbose=2)
    result = model.predict(x_test, batch_size=512)
    print(result,'\n\n')