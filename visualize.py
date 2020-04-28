from preprocessing import LoadData
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np

LData = LoadData()
x_data, y_data = LData.load_data()
y_recovery = list()
for i in range(int(y_data.size / 2)):
    if y_data[i][0] == 1.0:
        y_recovery.append(0)
    else:
        y_recovery.append(1)

x_data = pd.DataFrame(x_data, columns=[LData.columns])
y_data = pd.DataFrame(y_recovery, columns=['winner'])
total_data = pd.concat([x_data, y_data], axis=1)

scatter_matrix(total_data,
               alpha=0.5,
               figsize=(4, 4),
               diagonal='hist')
plt.show()
