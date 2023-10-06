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
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
#import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


Final_df = pd.read_excel("Final_df.xlsx")
Final_df_split = pd.read_excel("Final_df_split.xlsx")

X = Final_df.iloc[:,1:-1].values
y = Final_df.iloc[:, -1].values

columnName_list = []
varience_list = []
mean_list = []

columnName_list_split = []
varience_list_split = []
mean_list_split = []

for (columnName, columnData) in Final_df.iteritems():

    columnName_list.append(columnName)
    mean_dist = np.mean(columnData.values)
    mean_list.append(mean_dist)
    standard_dev = np.std(columnData.values)
    varience_list.append((standard_dev/mean_dist)*100)

for (columnName, columnData) in Final_df_split.iteritems():

    columnName_list_split.append(columnName)
    mean_dist = np.mean(columnData.values)
    mean_list_split.append(mean_dist)
    standard_dev = np.std(columnData.values)
    varience_list_split.append((standard_dev/mean_dist)*100)

columnName_list_split = columnName_list_split[0:len(columnName_list)]
varience_list_split = varience_list_split[0:len(varience_list)]

threshold = 0.4

print("\nVarience to mean threshold results : \n")

for i in range(len(columnName_list)):

    if(varience_list[i]/varience_list_split[i] > (1 + threshold) or varience_list[i]/varience_list_split[i] < (1 - threshold)):
        print(columnName_list[i] + " : " + str(varience_list[i]) + " , " + str(varience_list_split[i]))

print("\nMean threshold results : \n")

for i in range(len(columnName_list)):

    if(mean_list[i]/mean_list_split[i] > (1 + threshold) or mean_list[i]/mean_list_split[i] < (1 - threshold)):
        print(columnName_list[i] + " : " + str(mean_list[i]) + " , " + str(mean_list_split[i]))