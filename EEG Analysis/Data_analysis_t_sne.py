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

X = Final_df.iloc[:,1:-1].values
y = Final_df.iloc[:, -1].values

n_components = 2

# Setup grid
fig, ax = plt.subplots(1,6, figsize=(27,5))

# Loop
for i, p in enumerate( [5, 10, 15, 20, 30, 100] ):
  # Run t-SNE
  t_sne = TSNE(n_components=n_components,
              perplexity=p,
              random_state=12)
  
  # Calculate Matrix
  matrix_fit = t_sne.fit_transform(X)

  # Gather products and SNE transformation
  Final_df['sne_1'] = matrix_fit[:,0]
  Final_df['sne_2'] = matrix_fit[:,1]
  g = sns.scatterplot(data=Final_df, x='sne_1', y='sne_2', hue='Output', ax=ax[i])
  g.set_title(f'perplexity={p}')

plt.show()