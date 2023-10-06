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


Final_df = pd.read_excel("Final_df_split.xlsx")

X = Final_df.iloc[:,1:-1].values
y = Final_df.iloc[:, -1].values

Participants_count = len(Final_df.index)//10
print(Participants_count)

LR_scores = []
XGB_scores = []
KNN_scores = []
SVM_scores = []
KSVM_scores = []
NB_scores = []
DTC_scores = []
RFC_scores = []
Model_scores_list = []

X = pd.DataFrame(X)
y = pd.DataFrame(y)

# Outlier Detection

#min_percentile = 0.20
#max_percentile = 0.80
#X = cap_data(X, min_percentile, max_percentile)
#y = cap_data(y, min_percentile, max_percentile)

#X.to_excel("X_dataframe.xlsx")
#y.to_excel("y_dataframe.xlsx")

for i in range(Participants_count):

    #row_list = [5*i, 5*i + 1, 5*i + 2, 5*i + 3, 5*i + 4, 5*i + len(Final_df.index)//2,  5*i + (len(Final_df.index)//2) + 1 , 5*i + (len(Final_df.index)//2) + 2, 5*i + (len(Final_df.index)//2) + 3, 5*i + (len(Final_df.index)//2) + 4]
    row_list = list(range(5*i, 5*i + 5)) + list(range(5*i + len(Final_df.index)//2, 5*i + (len(Final_df.index)//2) + 5))

    X_train = X.drop(labels = row_list, axis=0)
    y_train = y.drop(labels = row_list, axis=0)
    X_test = X.iloc[row_list,:] 
    y_test = y.iloc[row_list,:] 

    # Feature Scaling

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Dimensionality Reduction

    # PCA - Principle Component Analysis

    #pca = PCA(n_components = 0.995)
    #X_train = pca.fit_transform(X_train)
    #X_test = pca.transform(X_test)
    #explained_variance = pca.explained_variance_ratio_
    #print(explained_variance)

    # LDA - Linear Discriminant Analysis

    #lda = LDA(n_components = 1)
    #X_train = lda.fit_transform(X_train,y_train)
    #X_test = lda.transform(X_test)

    # ML Models Accuracy Computation

    #print("X_train : ", X_train.shape)
    #print("y_train : ", y_train.shape)
    #print("X_test : ", X_test.shape)
    #print("y_test : ", y_test.shape)

    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    LR_classifier = LogisticRegression(random_state = 0)
    LR_classifier.fit(X_train, y_train)
    LR_scores.append(LR_classifier.score(X_test,y_test))

    XGB_classifier = XGBClassifier()
    XGB_classifier.fit(X_train, y_train)
    XGB_scores.append(XGB_classifier.score(X_test,y_test))

    KNN_classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
    KNN_classifier.fit(X_train, y_train)
    KNN_scores.append(KNN_classifier.score(X_test,y_test))

    SVM_classifier = SVC(kernel = 'linear', random_state = 0)
    SVM_classifier.fit(X_train, y_train)
    SVM_scores.append(SVM_classifier.score(X_test,y_test))

    KSVM_classifier = SVC(kernel = 'rbf', random_state = 0)
    KSVM_classifier.fit(X_train, y_train)
    KSVM_scores.append(KSVM_classifier.score(X_test,y_test))

    NB_classifier = GaussianNB()
    NB_classifier.fit(X_train, y_train)
    NB_scores.append(NB_classifier.score(X_test,y_test))

    DTC_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    DTC_classifier.fit(X_train, y_train)
    DTC_scores.append(DTC_classifier.score(X_test,y_test))

    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_train, y_train)
    RFC_scores.append(forest.score(X_test,y_test))


model_str = ["Logistic Regression","XG Boost", "KNN", "SVM", "Kernel SVM", "Naive Bayes", "Decision Trees Classifier", "Random Forest Classifier"]

Model_scores_list.append(LR_scores)
Model_scores_list.append(XGB_scores)
Model_scores_list.append(KNN_scores)
Model_scores_list.append(SVM_scores)
Model_scores_list.append(KSVM_scores)
Model_scores_list.append(NB_scores)
Model_scores_list.append(DTC_scores)
Model_scores_list.append(RFC_scores)

for i in range(len(model_str)):

    print(model_str[i] + " Score : " + str(mean(Model_scores_list[i])))

