import pathlib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pandas as pd
import numpy as np

train_file_path = ("./dataset/original/")
preprocessed_file_path = ("./dataset/preprocessed/")

train_file_challenger = pd.read_csv(train_file_path + "Challenger_Ranked_Games.csv").iloc[:, 1:]
train_file_grandmaster = pd.read_csv(train_file_path + "GrandMaster_Ranked_Games.csv").iloc[:, 1:]
train_file_master = pd.read_csv(train_file_path + "Master_Ranked_Games.csv").iloc[:, 1:]
train_file = pd.concat([train_file_challenger, train_file_grandmaster, train_file_master], axis=0)

cols_to_move = ['blueWins', 'redWins']
new_cols = np.hstack((train_file.columns.difference(cols_to_move), cols_to_move))

train_file = train_file.reindex(columns=new_cols)

