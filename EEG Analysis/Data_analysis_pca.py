import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from sklearn import svm
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from mpl_toolkits import mplot3d

def performMinMax(data_list):

    data_list_temp = []

    for i in range(len(data_list)):
        data_list_temp.append((data_list[i] - min(data_list)) / (max(data_list) - min(data_list)))

    return data_list_temp

Final_df = pd.read_excel("Final_df.xlsx")

X = Final_df.iloc[:,1:-1].values
y = Final_df.iloc[:, -1].values

# Dimensionality Reduction

# PCA - Principle Component Analysis

pca = PCA(n_components = 3)
X = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

pca_1 = []
pca_2 = []
pca_3 = []
result_list = y.tolist()

for i in X:
    pca_1.append(i[0])
    pca_2.append(i[1])
    pca_3.append(i[2])

pca_1 = performMinMax(pca_1)
pca_2 = performMinMax(pca_2)
pca_3 = performMinMax(pca_3)

xdata_pre = []
ydata_pre = []
zdata_pre = []
xdata_post = []
ydata_post = []
zdata_post = []

for i in range(len(pca_1)):

    if(result_list[i]==0):
        xdata_pre.append(pca_1[i])
        ydata_pre.append(pca_2[i])
        zdata_pre.append(pca_3[i])

    if(result_list[i]==1):
        xdata_post.append(pca_1[i])
        ydata_post.append(pca_2[i])
        zdata_post.append(pca_3[i])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xdata_pre, ydata_pre, zdata_pre, cmap='Reds')
ax.scatter3D(xdata_post, ydata_post, zdata_post, cmap='Greens')
ax.set_title("PCA 3D scatter plot")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")

plt.show()

