#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from logging import warning
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import VotingClassifier

# import models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# remove warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('Ecoli.csv')


# In[ ]:


print(df.isnull().values.any())
print(df.isnull().sum())


# In[ ]:


# drop the column with the target/unnecessary variables
X = df.drop(['Target (Col 107)'], axis=1)
y = df[['Target (Col 107)']]  # target

print(X.shape)
print(y.shape)
df.head()


# In[ ]:


# handle missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X)
X = imputer.transform(X)


# In[ ]:


# outliner detection
clf = IsolationForest(max_samples=100, random_state=1, contamination='auto')
clf.fit(X)


# In[ ]:


# Normalize the data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# In[ ]:


df_test = pd.read_csv('Ecoli_test.csv')
df_test.head()


# In[ ]:


print(df_test.isnull().values.any())
print(df_test.isnull().sum())


# In[ ]:


x_test = df_test
x_test.head()


# In[ ]:


# outliner detection
clf = IsolationForest(max_samples=100, random_state=1, contamination='auto')
clf.fit(x_test)


# In[ ]:


# Normalize the test data
scaler = StandardScaler()
scaler.fit(x_test)
x_test = scaler.transform(x_test)


# # KNN

# In[24]:


# knn model
knn = KNeighborsClassifier()
knn_params = {'n_neighbors': np.arange(1, 30)}
knn_grid = GridSearchCV(knn, knn_params, cv=10, scoring=[
                        'f1', 'accuracy'], refit='f1')
knn_grid.fit(X, y)
print(knn_grid.best_params_)
print(knn_grid.best_score_)
print(knn_grid.best_estimator_)
knn_grid.best_estimator_.fit(X, y)
y_pred_knn = knn_grid.best_estimator_.predict(x_test)
print(y_pred_knn)

accuracy_result = knn_grid.cv_results_[
    'mean_test_accuracy'][knn_grid.best_index_]
f1_result = knn_grid.cv_results_['mean_test_f1'][knn_grid.best_index_]

results = [round(accuracy_result, 3), round(f1_result, 3)]
print(results)


# # Decision Tree

# In[30]:


# decision tree model
dt = DecisionTreeClassifier()
dt_params = {'max_depth': np.arange(1, 30), 'criterion': ['gini', 'entropy']}
dt_grid = GridSearchCV(dt, dt_params, cv=10, scoring=[
                       'f1', 'accuracy'], refit='f1', n_jobs=3)
dt_grid.fit(X, y)
print(dt_grid.best_params_)
print(dt_grid.best_score_)
print(dt_grid.best_estimator_)

dt_grid.best_estimator_.fit(X, y)
y_pred_dt = dt_grid.best_estimator_.predict(x_test)
print(y_pred_dt)

accuracy_result = dt_grid.cv_results_[
    'mean_test_accuracy'][dt_grid.best_index_]
f1_result = dt_grid.cv_results_['mean_test_f1'][dt_grid.best_index_]

results = [round(accuracy_result, 3), round(f1_result, 3)]
print(results)

with open('s4415462.csv', mode='w', newline='') as resultFile:
    fullWriter = csv.writer(resultFile, delimiter=',', quotechar='"',
                            lineterminator=',\r\n', quoting=csv.QUOTE_MINIMAL)
    fullWriter.writerows(map(lambda x: [int(x)], y_pred_dt))
with open('s4415462.csv', mode='a', newline='') as resultFile:
    fullWriter = csv.writer(resultFile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    fullWriter.writerow(results)


# ## Random Forest

# In[31]:


# random forest model
rf = RandomForestClassifier()
rf_params = {'n_estimators': np.arange(1, 30)}
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring=[
                       'f1', 'accuracy'], refit='f1')
rf_grid.fit(X, y)
print(rf_grid.best_params_)
print(rf_grid.best_score_)
print(rf_grid.best_estimator_)
rf_grid.best_estimator_.fit(X, y)

y_pred_rf = rf_grid.best_estimator_.predict(x_test)
print(y_pred_rf)

accuracy_result = rf_grid.cv_results_[
    'mean_test_accuracy'][rf_grid.best_index_]
f1_result = rf_grid.cv_results_['mean_test_f1'][rf_grid.best_index_]

results = [round(accuracy_result, 3), round(f1_result, 3)]
print(results)


# ##  naïve bayes

# In[32]:


# naive bayes model
nb = GaussianNB()
nb_params = {'var_smoothing': np.arange(1, 30)}
nb_grid = GridSearchCV(nb, nb_params, cv=10, scoring=[
                       'f1', 'accuracy'], refit='f1')
nb_grid.fit(X, y)
print(nb_grid.best_params_)
print(nb_grid.best_score_)
print(nb_grid.best_estimator_)
nb_grid.best_estimator_.fit(X, y)

y_pred_nb = nb_grid.best_estimator_.predict(x_test)
print(y_pred_nb)

accuracy_result = nb_grid.cv_results_[
    'mean_test_accuracy'][nb_grid.best_index_]
f1_result = nb_grid.cv_results_['mean_test_f1'][nb_grid.best_index_]

results = [round(accuracy_result, 3), round(f1_result, 3)]
print(results)


# In[33]:


# hard voting for classifiers
voting_clf = VotingClassifier(estimators=[('knn', knn_grid.best_estimator_), ('dt', dt_grid.best_estimator_), (
    'rf', rf_grid.best_estimator_), ('nb', nb_grid.best_estimator_)], voting='hard')
voting_clf.fit(X, y)
y_pred_voting = voting_clf.predict(x_test)
print(y_pred_voting)

accuracy_result = accuracy_score(y, voting_clf.predict(X))
f1_result = f1_score(y, voting_clf.predict(X), average='binary')

results = [round(accuracy_result, 3), round(f1_result, 3)]
print(results)
