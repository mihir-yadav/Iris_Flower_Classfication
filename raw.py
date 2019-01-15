#!/usr/bin/env python
# coding: utf-8

# #          Iris Flowers Classification - Supervised Machine Learning 

# In[ ]:


import sys
import scipy
import numpy 
import pandas
import sklearn


# In[3]:


# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# # We are going to use the Fisher's Iris data set for training, validation and testing of our algorithm.

# In[4]:


# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)


# # We shall begin with the analysis and visualization of our data.

# In[5]:


# shape
print(dataset.shape)


# In[6]:


# head
print(dataset.head(20))


# In[7]:


# descriptions
print(dataset.describe())


# In[8]:


# class distribution
print(dataset.groupby('class').size())


# In[9]:


# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# In[10]:


import warnings
warnings.simplefilter("ignore")
# histograms
dataset.hist()
plt.show()


# In[11]:


# scatter plot matrix
scatter_matrix(dataset)
plt.show()


# # We shall split the dataset into 2 parts in the ratio 4:1 using the former to train each model and the latter to test the best model.

# In[17]:


# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
test_size = 0.20
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)


# In[18]:


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


# # We are not sure which algorithm would suit best, so we choose the one which gives maximum accuracy on the validation set.

# In[14]:


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# In[15]:


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# # It is now evident that SVM gives the largest estimated accuracy score on validation set. So we use it to make predictions on test data.

# In[21]:


# Make predictions on test dataset
svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))


# # We can see that the accuracy is more than 93%.The confusion matrix provides an indication of the three errors made.

# # Finally, the classification report provides a breakdown of each class by precision, recall, f1-score and support showing excellent results.
