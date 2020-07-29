#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import pickle
import numpy as np
import sys
from classifier import Classifier


# In[2]:


rep_id = int(sys.argv[1])
method_id = int(sys.argv[2])
MFE_model_id = int(sys.argv[3])
CNN_model_id = int(sys.argv[4])
training_size_id = int(sys.argv[5])


# In[3]:


method_list = ['MFE', 'CNN', 'Joint', 'SE']
model_list = [['RF', 'FNN', 'SVM'], [32, 64, 96, 128]]
training_size_list = [500, 5000, 50000, 162946]

# In[ ]:


method = method_list[method_id]
training_size = training_size_list[training_size_id]

if method == 'MFE':
    model = model_list[method_id][MFE_model_id]
else:
    model = model_list[1][CNN_model_id]
    
clf = Classifier()
(macro, micro, cm), prob = clf.classifier(rep_id, method, model, training_size)

filename_f1 = './result/{0}_{1}_{2}_{3}_{4}.csv'.format(rep_id, training_size, method, model, 'f1')
filename_cm = './result/{0}_{1}_{2}_{3}_{4}.csv'.format(rep_id, training_size, method, model, 'cm')
filename_prob_test = './result/{0}_{1}_{2}_{3}_{4}.pickle'.format(rep_id, training_size, method, model, 'prob_test')
filename_prob_train = './result/{0}_{1}_{2}_{3}_{4}.pickle'.format(rep_id, training_size, method, model, 'prob_train')

with open(filename_f1, 'w') as f:
    w = csv.writer(f)
    w.writerow([macro, micro])
    
with open(filename_cm, 'w') as f:
    w = csv.writer(f)
    w.writerows(cm)
if method =='CNN' or model =='FNN':
    with open(filename_prob_train, 'wb') as f:
        pickle.dump(prob[0], f)
    with open(filename_prob_test, 'wb') as f:
        pickle.dump(prob[1], f)
    

