#!/usr/bin/env python
# coding: utf-8

# # EpLogic - Experiment #4

# ## Common imports

# In[4]:


print()

import warnings; warnings.simplefilter('ignore')

import pandas                  as pd
import numpy                   as np
import sklearn.model_selection as ms
import sklearn.metrics         as mt

from imblearn.under_sampling   import RandomUnderSampler
from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

print('All fine!')


# ## Retrieving the datasets

# In[5]:


print()

instances_pe = pd.read_csv('./datasets/auto-epa/20200130/pe-dataset.csv')
print('PE dataset:', instances_pe.shape)

instances_sp = pd.read_csv('./datasets/auto-epa/20200130/sp-dataset.csv')
print('SP dataset:', instances_sp.shape)


# ## Cleaning datasets

# In[6]:


print()

cln_instances_pe = instances_pe.drop(columns=['panel_info', 'panel_eplet', 'locus_abc', 'locus_dp', 'locus_dq', 'locus_dr'])
print('PE cleaned dataset:', cln_instances_pe.shape)

cln_instances_sp = instances_sp.drop(columns=['panel_info', 'panel_eplet'])
print('SP cleaned dataset:', cln_instances_sp.shape)


# ## Preparing train/validation instances

# In[7]:


print()

imb_train_labels_pe = np.array(cln_instances_pe['reactive'])
imb_instances_pe = cln_instances_pe.drop(columns=['reactive'])

rus = RandomUnderSampler()

train_instances_pe, train_labels_pe = rus.fit_sample(imb_instances_pe, imb_train_labels_pe)

print('Train labels (PE)', train_instances_pe.shape)
print('Train instances (PE):', train_labels_pe.shape)

print()

imb_train_labels_sp = np.array(cln_instances_sp['reactive'])
imb_instances_sp = cln_instances_sp.drop(columns=['reactive'])

train_instances_sp, train_labels_sp = rus.fit_sample(imb_instances_sp, imb_train_labels_sp)

print('Train labels (SP)', train_instances_sp.shape)
print('Train instances (SP):', train_labels_sp.shape)


# ## Training the model and validating it

# In[17]:


clf = RandomForestClassifier()
cv = 5
scoring = ['roc_auc', 'accuracy']

print()
print('Parameter names currently in use by RF:\n')
print(clf.get_params())

param_grid = {
  'n_estimators': [100, 200, 400],
  'criterion': ['gini', 'entropy'],
  'min_samples_split': [2, 4, 8],
  'min_samples_leaf': [1, 2, 4],
  'max_features': [None, 'auto', 'sqrt', 'log2']
}

print('\nPE ---')

params_search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, n_iter=cv, scoring='roc_auc', n_jobs=3, cv=(cv*cv))
params_search.fit(train_instances_pe, train_labels_pe)

print()
print('Best parameters:\n')
print(params_search.best_params_)

predicted_labels_pe = ms.cross_val_predict(params_search.best_estimator_, train_instances_pe, train_labels_pe, n_jobs=3, cv=cv)

print()
print('First step:\n')
print('- Confusion matrix (TN FP FN TP):', mt.confusion_matrix(train_labels_pe, predicted_labels_pe).ravel())
print("- AUC-ROC: %0.2f" % (mt.roc_auc_score(train_labels_pe, predicted_labels_pe)*100))
print("- Accuracy: %0.2f" % (mt.accuracy_score(train_labels_pe, predicted_labels_pe)*100))

scores = ms.cross_validate(params_search.best_estimator_, train_instances_pe, train_labels_pe, scoring=scoring, n_jobs=3, cv=cv)

print()
print('Second step:\n')
print("- AUC-ROC:  %0.2f (+/- %0.2f)" % (scores['test_roc_auc'].mean()*100, scores['test_roc_auc'].std()*100))
print("- Accuracy: %0.2f (+/- %0.2f)" % (scores['test_accuracy'].mean()*100, scores['test_accuracy'].std()*100))


#---


print('\nSP ---')

params_search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, n_iter=cv, scoring='roc_auc', n_jobs=3, cv=(cv*cv))
params_search.fit(train_instances_sp, train_labels_sp)

print()
print('Best parameters:\n')
print(params_search.best_params_)

predicted_labels_sp = ms.cross_val_predict(params_search.best_estimator_, train_instances_sp, train_labels_sp, cv=cv, n_jobs=-1)

print()
print('First step:\n')
print('- Confusion matrix (TN FP FN TP):', mt.confusion_matrix(train_labels_sp, predicted_labels_sp).ravel())
print("- AUC-ROC: %0.2f" % (mt.roc_auc_score(train_labels_sp, predicted_labels_sp)*100))
print("- Accuracy: %0.2f" % (mt.accuracy_score(train_labels_sp, predicted_labels_sp)*100))

scores = ms.cross_validate(params_search.best_estimator_, train_instances_sp, train_labels_sp, cv=cv, n_jobs=-1, scoring=scoring)

print()
print('Second step:\n')
print("- AUC-ROC:  %0.2f (+/- %0.2f)" % (scores['test_roc_auc'].mean()*100, scores['test_roc_auc'].std()*100))
print("- Accuracy: %0.2f (+/- %0.2f)" % (scores['test_accuracy'].mean()*100, scores['test_accuracy'].std()*100))


# In[ ]:




