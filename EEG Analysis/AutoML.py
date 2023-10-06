from flaml import AutoML
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter
from numpy import mean,absolute,sqrt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

Final_df = pd.read_excel("Final_df_split.xlsx")

# print(Final_df)

X = Final_df.iloc[:,1:-1].values
y = Final_df.iloc[:, -1].values

# Initialize an AutoML instance
automl = AutoML()
# Specify automl goal and constraint
automl_settings = {
    "time_budget": 60,  # in seconds
    "metric": 'accuracy',
    "task": 'classification',
    "log_file_name": "EEG_analysis_automl.log",
}

# Train with labeled input data
automl.fit(X_train=X, y_train=y,**automl_settings)
# Print the best model
print("\n\nTHIS IS THE BEST MODEL AS OF NOW : \n\n",automl.model.estimator)

cv = LeaveOneOut()
scores = cross_val_score(automl, X, y, scoring='neg_mean_absolute_error',cv=cv, n_jobs=-1)
print("--> AutoML Model (LOOCV) : ", mean(absolute(scores)))