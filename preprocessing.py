import pathlib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

train_file_path = ("./dataset/original/")
preprocessed_file_path = ("./dataset/preprocessed/")

train_file_challenger = pd.read_csv(train_file_path + "Challenger_Ranked_Games.csv").iloc[:, 1:]
train_file_grandmaster = pd.read_csv(train_file_path + "GrandMaster_Ranked_Games.csv").iloc[:, 1:]
train_file_master = pd.read_csv(train_file_path + "Master_Ranked_Games.csv").iloc[:, 1:]
train_file = pd.concat([train_file_challenger, train_file_grandmaster, train_file_master], axis=0).reset_index(
    drop=True)

num_of_features = 47
scaler = MinMaxScaler()
str_encoder = OneHotEncoder()
cols_to_remove = ['blueWins', 'redWins']

winner = list()
for i in train_file.index:
    if train_file['blueWins'].iloc[i] == 0:
        winner.append('blue')
    else:
        winner.append('red')
winner = pd.DataFrame(winner, columns=['winner'])
train_file = pd.concat([train_file, winner], axis=1)
train_file = train_file.drop(cols_to_remove, axis=1)

x_data = train_file.iloc[:, 0:-1]
y_data = np.array(train_file.iloc[:, -1]).reshape(-1, 1)

print("Shape of x_data", "\n", x_data.shape)
print("Shape of y_data", "\n", y_data.shape)

scaler.fit(x_data)
x_data_scaled = scaler.transform(x_data)

str_encoder.fit(y_data)
y_data_onehot = str_encoder.transform(y_data).toarray()
y_data_recovery = np.argmax(y_data_onehot, axis=1).reshape(-1, 1)


class LoadData:
    def __init__(self):
        self.x_data = x_data_scaled
        self.y_data = y_data_onehot

    def load_data(self):
        return self.x_data, self.y_data
