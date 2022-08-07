#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris=load_iris()
print(iris.DESCR)


# In[2]:


import pandas as pd 

data=pd.DataFrame(iris.data)
data.head()


# In[3]:


data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
data.head()


# In[5]:


target = pd.DataFrame(iris.target)
target = target.rename(columns={0:'target'})
target.head()


# In[6]:


df = pd.concat([data,target],axis=1)
df.head()


# In[7]:


df.dtypes


# In[8]:


df.isnull().sum()


# In[9]:


df.describe()


# In[10]:


import seaborn as sns 
import matplotlib.pyplot as plt

sns.heatmap(df.corr(),annot=True)


# In[19]:


import matplotlib.pyplot as plt

x_index = 0
y_index=1

formatter = plt.FuncFormatter(lambda i,*args: 
                              iris.target_names[int(i)])

plt.figure(figsize=(5,4))
plt.scatter(iris.data[:,x_index],iris.data[:,y_index],
            c=iris.target)
plt.colorbar(ticks=[0,1,2],format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])

plt.tight_layout()
plt.show()


# In[20]:


X=df.copy()
y=X.pop('target')


# In[23]:


from sklearn.model_selection import train_test_split

X=df.copy()
y=X.pop('target')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify = y)

'''
by stratifying on y we assure that the different classes are represented proportionally to the amount in the total data (this makes sure that all of class 1 is not in the test group only)
'''


# In[25]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

df.target.value_counts(normalize= True)


# In[27]:


from sklearn.linear_model import LogisticRegression
import numpy as np

#create the model instance
model = LogisticRegression()
#fit the model on the training data
model.fit(X_train, y_train)
#the score, or accuracy of the model
model.score(X_test, y_test)
# Output = 0.9666666666666667
#the test score is already very high, but we can use the cross validated score to ensure the model's strength 
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=10)
print(np.mean(scores))
# Output = 0.9499999999999998


# In[28]:


df_coef = pd.DataFrame(model.coef_, columns=X_train.columns)
df_coef


# In[29]:


predictions = model.predict(X_test)
#compare predicted values with the actual scores
compare_df = pd.DataFrame({'actual': y_test, 'predicted': predictions})
compare_df = compare_df.reset_index(drop = True)
compare_df


# In[30]:


from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(y_test, predictions, labels=[2, 1, 0]),index=[2, 1, 0], columns=[2, 1, 0])


# In[31]:


from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# In[32]:


probs = model.predict_proba(X_test)
#put the probabilities into a dataframe for easier viewing
Y_pp = pd.DataFrame(model.predict_proba(X_test), 
             columns=['class_0_pp', 'class_1_pp', 'class_2_pp'])
Y_pp.head()


# In[ ]:




