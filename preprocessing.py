import pathlib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tensorflow as tf
import pandas as pd
import numpy as np


train_file_path = ("./dataset/original/")
preprocessed_file_path = ("./dataset/preprocessed/")

train_file_challenger = pd.read_csv(train_file_path + "Challenger_Ranked_Games.csv").iloc[:, 1:]
train_file_grandmaster = pd.read_csv(train_file_path + "GrandMaster_Ranked_Games.csv").iloc[:, 1:]
train_file_master = pd.read_csv(train_file_path + "Master_Ranked_Games.csv").iloc[:, 1:]
train_file = pd.concat([train_file_challenger, train_file_grandmaster, train_file_master], axis=0).reset_index(drop=True)

num_of_features = 47
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

num_pipeline = Pipeline([

])

str_pipeline = Pipeline([

])

full_pipeline = Pipeline([

])

