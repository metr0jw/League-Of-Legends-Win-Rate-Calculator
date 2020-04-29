from preprocessing import LoadData
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np

LData = LoadData()
x_processed, y_processed, x_origin, y_origin = LData.load_data()

y_recovery = list()
for i in range(len(y_processed)):
    if y_processed[i][0] is 1.0:
        y_recovery.append('0')
    else:
        y_recovery.append('1')

x_data = x_origin
y_data = pd.DataFrame(y_recovery, columns=['winner'])
total_data = pd.concat([x_data, y_data], axis=1)
