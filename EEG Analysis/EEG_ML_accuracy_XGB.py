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


def cap_data(df,min_percentile,max_percentile):

    for col in df.columns:
    
        print("capping the ",col)
    
        if (((df[col].dtype)=='float64') | ((df[col].dtype)=='int64')):
            percentiles = df[col].quantile([min_percentile,max_percentile]).values
            df[col][df[col] <= percentiles[0]] = percentiles[0]
            df[col][df[col] >= percentiles[1]] = percentiles[1]
    
        else:
            df[col]=df[col]
    
    return df


Final_df = pd.read_excel("Final_df.xlsx")

X = Final_df.iloc[:,1:-1].values
y = Final_df.iloc[:, -1].values

# Outlier Detection

#min_percentile = 0.20
#max_percentile = 0.80
#X = cap_data(X, min_percentile, max_percentile)
#y = cap_data(y, min_percentile, max_percentile)

Participants_count = len(Final_df.index)//2

XGB_scores = []

X = pd.DataFrame(X)
y = pd.DataFrame(y)

params = { 'max_depth': [3,6,10],
           'learning_rate': [0.01, 0.05, 0.1],
           'n_estimators': [100, 500, 1000],
           'colsample_bytree': [0.3, 0.7]}


for i in range(Participants_count):

    X_train = X.drop(labels = [i, i + Participants_count], axis=0)
    y_train = y.drop(labels = [i, i + Participants_count], axis=0)
    X_test = X.iloc[[i, i + Participants_count],:] 
    y_test = y.iloc[[i, i + Participants_count],:] 

    if(i==0):

        X_train.to_excel("X_dataframe.xlsx")
        y_train.to_excel("y_dataframe.xlsx")

    # Feature Scaling

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # ML Models Accuracy Computation

    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    XGB_classifier = XGBClassifier()
    XGB_classifier.fit(X_train, y_train)
    XGB_scores.append(XGB_classifier.score(X_test,y_test))
    #plt.bar(range(len(XGB_classifier.feature_importances_)), XGB_classifier.feature_importances_)
    #plt.show()

print("XGB Score : " + str(mean(XGB_scores)))
print(XGB_scores)

