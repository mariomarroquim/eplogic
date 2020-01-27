#!/usr/bin/env python
# coding: utf-8

# # EpLogic - Experiment 1

# ## Common imports

# In[56]:


print()

import warnings; warnings.simplefilter('ignore')

import pandas                  as pd
import numpy                   as np
import sklearn.model_selection as ms
import sklearn.metrics         as mt

from imblearn.under_sampling   import RandomUnderSampler
from sklearn.ensemble          import RandomForestClassifier

print('All fine!')


# ## Retrieving datasets

# In[57]:


print()

instances_pe = pd.read_csv('./datasets/pe-dataset.csv')
print('PE dataset:', instances_pe.shape)

instances_sp = pd.read_csv('./datasets/sp-dataset.csv')
print('SP dataset:', instances_sp.shape)


# ## Cleaning datasets

# In[58]:


print()

cln_instances_pe = instances_pe.drop(columns=['panel_info', 'panel_eplet'])
print('PE cleaned dataset:', cln_instances_pe.shape)

cln_instances_sp = instances_sp.drop(columns=['panel_info', 'panel_eplet'])
print('SP cleaned dataset:', cln_instances_sp.shape)


# ## Exploring the PE dataset

# In[59]:


cln_instances_pe.head(3)


# In[60]:


print()

eplets_pe_count = cln_instances_pe.reactive.value_counts()

print('Non reactive:', eplets_pe_count[0])
print('Reactive:    ', eplets_pe_count[1])
print('Proportion:  ', int(eplets_pe_count[0] / eplets_pe_count[1]), ': 1')


# ## Exploring the SP dataset

# In[61]:


cln_instances_sp.head(3)


# In[62]:


print()

eplets_sp_count = cln_instances_sp.reactive.value_counts()

print('Non reactive:', eplets_sp_count[0])
print('Reactive:    ', eplets_sp_count[1])
print('Proportion:  ', int(eplets_sp_count[0] / eplets_sp_count[1]), ': 1')


# ## Preparing train/validation instances

# In[63]:


print()

imb_train_labels_pe = np.array(cln_instances_pe['reactive'])
imb_instances_pe = cln_instances_pe.drop(columns=['reactive'])

rus = RandomUnderSampler()
train_instances_pe, train_labels_pe = rus.fit_sample(imb_instances_pe, imb_train_labels_pe)

print('Train labels (PE)', train_instances_pe.shape)
print('Train instances (PE):', train_labels_pe.shape)

print()

validation_labels_sp = np.array(cln_instances_sp['reactive'])
validation_instances_sp = cln_instances_sp.drop(columns=['reactive'])

print('Validation labels (SP)', validation_instances_sp.shape)
print('Validation instances (SP):', validation_labels_sp.shape)


# ## Training the model and validating it

# In[64]:


clf = RandomForestClassifier(n_estimators=100)

clf.fit(train_instances_pe, train_labels_pe)

feature_importances = pd.DataFrame(clf.feature_importances_,
                                   index = cln_instances_pe.drop(columns=['reactive']).columns,
                                   columns = ['importance']).sort_values('importance', ascending=False)

print()
print('Variable importance:\n\n', feature_importances)

predicted_labels_sp = clf.predict(validation_instances_sp)

print()
print('Confusion matrix:\n\n', mt.confusion_matrix(validation_labels_sp, predicted_labels_sp))

print()
print('Validation metrics:\n')
print(' - Accuracy: %0.2f' % (mt.accuracy_score(validation_labels_sp, predicted_labels_sp) * 100))
print(' - AUC-ROC:  %0.2f' % (mt.roc_auc_score(validation_labels_sp, predicted_labels_sp) * 100))


# In[ ]:




