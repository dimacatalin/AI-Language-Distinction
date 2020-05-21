#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import f1_score, classification_report
import pdb
import csv
from sklearn.metrics import confusion_matrix
import seaborn as sn


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


trainSamples = np.genfromtxt(r'C:\Users\DMO\Desktop\IAProject\train_samples.txt', encoding='utf-8', dtype=None, delimiter='\t', names=('col1','col2'),comments=None)
trainLabels = np.genfromtxt(r'C:\Users\DMO\Desktop\IAProject\train_labels.txt', encoding='utf-8', dtype=None, delimiter='\t', names=('col1','col2'),comments=None)

testSamples = np.genfromtxt(r'C:\Users\DMO\Desktop\IAProject\test_samples.txt', encoding='utf-8', dtype=None, delimiter='\t', names=('col1','col2'),comments=None)

validationSamples = np.genfromtxt(r'C:\Users\DMO\Desktop\IAProject\validation_samples.txt', encoding='utf-8', dtype=None, delimiter='\t', names=('col1','col2'),comments=None)
validationLabels = np.genfromtxt(r'C:\Users\DMO\Desktop\IAProject\validation_labels.txt', encoding='utf-8', dtype=None, delimiter='\t', names=('col1','col2'),comments=None)


# In[4]:


len(trainLabels),len(trainSamples),len(testSamples),len(validationSamples),len(validationSamples)


# In[5]:


class BagOfWords:

    def __init__(self):
        self.vocab = {}
        self.words = []

    def build_vocabulary(self, train_data):
        for sentence in train_data:
            for word in sentence.split(" "):
                if word not in self.vocab:
                    self.vocab[word] = len(self.words)
                    self.words.append(word)
        return self.words

    def get_features(self, data):
        result = np.zeros((data.shape[0], len(self.words)))
        for idx, sentence in enumerate(data):
            for word in sentence.split(" "):
                if word in self.vocab:
                    result[idx, self.vocab[word]] += 1
        return result


# In[6]:


bow_model = BagOfWords()
bow_model.build_vocabulary(trainSamples["col2"]) 


# In[7]:


def compute_accuracy(gt_labels, predicted_labels):
    accuracy = np.sum(predicted_labels == gt_labels) / len(predicted_labels)
    return accuracy

def normalize_data(train_data, test_data, type=None):
    scaler = None
    if type == 'standard':
        scaler = preprocessing.StandardScaler()

    elif type == 'min_max':
        scaler = preprocessing.MinMaxScaler()

    elif type == 'l1':
        scaler = preprocessing.Normalizer(norm='l1')

    elif type == 'l2':
        scaler = preprocessing.Normalizer(norm='l2')

    if scaler is not None:
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_test_data = scaler.transform(test_data) 
        return (scaled_train_data, scaled_test_data)
    else:
        print("No scaling was performed. Raw data is returned.")
        return (train_data, test_data)


# In[8]:


train_features = bow_model.get_features(trainSamples["col2"])
train_features[0]


# In[9]:


test_features = bow_model.get_features(validationSamples["col2"]) #era testSamples
test_features[0]


# In[10]:


scaled_train_data, scaled_test_data = normalize_data(train_features, test_features, type='l2')

svm_model = svm.SVC(C=100, kernel='linear')
svm_model.fit(scaled_train_data, trainLabels["col2"])


# In[11]:


predicted_labels_svm = svm_model.predict(scaled_test_data) 
predicted_labels_svm


# In[12]:


len(predicted_labels_svm)


# In[ ]:


with open('rezultat.csv','w',newline='') as fin:
    fieldnames = ['id','label']
    f = csv.DictWriter(fin, fieldnames = fieldnames)
    
    f.writeheader()
    
    for i in range(0,len(predicted_labels_svm)):
        f.writerow({'id' : testSamples["col1"][i],'label' : predicted_labels_svm[i]})


# In[22]:


model_accuracy_svm = compute_accuracy(np.asarray(validationLabels["col2"]), predicted_labels_svm)
print("Accuracy: ", model_accuracy_svm * 100)


# In[24]:


print('f1 score', f1_score(np.asarray(validationLabels["col2"]), predicted_labels_svm))


# In[36]:


confusion_matrix(np.asarray(validationLabels["col2"]), predicted_labels_svm)


# In[35]:


bz = confusion_matrix(np.asarray(validationLabels["col2"]), predicted_labels_svm)
plt.figure(figsize = (10,7))
sn.heatmap(bz, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Correct')


# In[ ]:




