import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

def performMinMax(data_list):

    data_list_temp = []

    for i in range(len(data_list)):
        data_list_temp.append((data_list[i] - data_list.min()) / (data_list.max() - data_list.min()))

    return data_list_temp


Final_df = pd.read_excel("Final_df_split.xlsx")

X = Final_df.iloc[:,1:-1].values
y = Final_df.iloc[:, -1].values

columnName_list = []
varience_list = []

for (columnName, columnData) in Final_df.iteritems():

    columnName_list.append(columnName)
    mean_dist = np.mean(columnData.values)
    standard_dev = np.std(columnData.values)
    varience_list.append((standard_dev/mean_dist)*100)

    print('\nColumn Name : ', columnName)
    print("Column Mean : ",mean_dist)
    print('Column Standard Deviation : ', standard_dev)
    print("Percentage of std dev to mean : ", (standard_dev/mean_dist)*100)
    print("\n")

threshold_val = 200
feature_num = 0

for i in range(len(columnName_list)):

    if(varience_list[i] > threshold_val):

        print(columnName_list[i] + " : " + str(varience_list[i]))
        feature_num += 1

print("Total number of features exhibiting above threshold varience : ", str(feature_num))

XGB_importance_list = ["Delta_value_2","Alpha_frequency_9","Alpha_frequency_10"]
XGB_data = []

for (columnName, columnData) in Final_df.iteritems():
    if columnName in XGB_importance_list:
        XGB_data.append(columnData)

fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for three-dimensional scattered points

xdata = XGB_data[0]
ydata = XGB_data[1]
zdata = XGB_data[2]
result_list = y.tolist()

xdata = performMinMax(xdata)
ydata = performMinMax(ydata)
zdata = performMinMax(zdata)

xdata_pre = []
ydata_pre = []
zdata_pre = []
xdata_post = []
ydata_post = []
zdata_post = []

for i in range(len(xdata)):

    if(result_list[i]==0):
        xdata_pre.append(xdata[i])
        ydata_pre.append(ydata[i])
        zdata_pre.append(zdata[i])

    if(result_list[i]==1):
        xdata_post.append(xdata[i])
        ydata_post.append(ydata[i])
        zdata_post.append(zdata[i])

ax.scatter3D(xdata_pre, ydata_pre, zdata_pre, cmap='Reds')
ax.scatter3D(xdata_post, ydata_post, zdata_post, cmap='Greens')
ax.set_title("XGBoost top features 3D plot")
ax.set_xlabel("Delta val 2")
ax.set_ylabel("Alpha freq 9")
ax.set_zlabel("Alpha freq 10")

plt.show()
