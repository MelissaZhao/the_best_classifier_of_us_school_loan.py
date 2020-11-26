#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load Data

# In[2]:


get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# In[3]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[4]:


df.shape


# # Data Pre-processing

# In[5]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# In[8]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# In[9]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# In[10]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# In[11]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# In[73]:


df['education'].replace(to_replace=['High School or Below','college','Bechalor','Master or Above'], value=[0,1,2,3], inplace=True)
df.head()


# In[18]:


Feature = df[['Principal','terms','age','education','Gender','weekend']]
Feature.head()


# In[19]:


X = Feature
X[0:5]


# In[20]:


y = df['loan_status'].values
y[0:5]


# In[21]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Classification

# # 1. K Nearest Neighbor (KNN)

# In[23]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=5)
X_train = preprocessing.StandardScaler().fit(X_train).transform(X_train)
X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[25]:


mean_acc=np.zeros(50)
std_acc = np.zeros(50)
for n in range(1,50):
    knnmodel=KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
    y_pred=knnmodel.predict(X_test)
    mean_acc[n-1]=metrics.accuracy_score(y_test,y_pred)
    std_acc[n-1]=np.std(y_pred==y_test)/np.sqrt(y_pred.shape[0])
    
plt.plot(range(1,51),mean_acc,'g')
plt.fill_between(range(1,51),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


# In[26]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)


# # 2. Decision Tree (DT)

# In[27]:


get_ipython().system('conda install pydotplus')


# In[28]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree


# In[29]:


dt= DecisionTreeClassifier(criterion="entropy", max_depth = 5)
dt.fit(X_train,y_train)


# In[35]:


dot_data = StringIO()
filename = "dt.png"
featureNames = df.columns[2: 8]
targetNames = df["loan_status"].unique().tolist()
out=tree.export_graphviz(dt,feature_names= featureNames, out_file=dot_data, filled=True, special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')


# In[36]:


y_pred=dt.predict(X_test)
TreeAccuracy=metrics.accuracy_score(y_test,y_pred)
TreeAccuracy


# # 3. Logistic Regression (LR)

# In[37]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
lr=LogisticRegression(C=0.01,solver='liblinear').fit(X_train,y_train)
lr


# In[40]:


y_pred = lr.predict(X_test)
y_pred[0:5]


# In[41]:


y_pred_prob = lr.predict_proba(X_test)


# In[53]:


metrics.accuracy_score(y_test,y_pred)


# # 4. Support Vector Machine (SVM)

# In[47]:


from sklearn import svm
svm=svm.SVC(kernel='rbf')
svm.fit(X_train, y_train) 


# In[50]:


y_pred = svm.predict(X_test)
y_pred[0:5]


# In[52]:


metrics.accuracy_score(y_test,y_pred)


# # Evaluation

# In[54]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# In[114]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# In[75]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[76]:


test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])


# In[77]:


test_df['weekend'] = (test_df['effective_date'].dt.dayofweek).apply(lambda x: 1 if (x>3)  else 0)


# In[78]:


test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)


# In[79]:


test_df['education'].replace(to_replace=['High School or Below','college','Bechalor','Master or Above'], value=[0,1,2,3], inplace=True)
df.head()


# In[80]:


test_Feature = df[['Principal','terms','age','education','Gender','weekend']]
test_Feature.head()


# In[81]:


y_testset = test_df['loan_status'].values

X_testset = test_Feature

X_testset = preprocessing.StandardScaler().fit(X_testset).transform(X_testset)
X_testset[0:5]


# # 1. Evaluation KNN

# In[89]:


y_pred_knn=knnmodel.predict(X_testset)
jacc_knn=jaccard_similarity_score(y_testset,y_pred_knn)
jacc_knn


# In[91]:


f1_knn = f1_score(y_testset, y_pred_knn, average='weighted')
f1_knn


# In[93]:


logl_knn = 'NA'


# # 2. Evaluation DT

# In[95]:


y_pred_dt = dt.predict(X_testset)
jacc_dt= jaccard_similarity_score(y_testset, y_pred_knn)
jacc_dt


# In[96]:


f1_dt = f1_score(y_testset, y_pred_dt, average='weighted')
f1_dt


# In[97]:


logl_dt = 'NA'


# # 3. Evaluation LR

# In[99]:


y_pred_lr = lr.predict(X_testset)
jacc_lr = jaccard_similarity_score(y_testset, y_pred_lr )
jacc_lr


# In[100]:


f1_lr = f1_score(y_testset, y_pred_lr, average='weighted')
f1_lr


# In[106]:


y_pred_lr_prob = lr.predict_proba(X_testset)
logl_lr = log_loss(y_testset, y_pred_lr_prob)
logl_lr


# # 4. Evaluation SVM

# In[103]:


y_pred_svm = svm.predict(X_testset)
jacc_svm = jaccard_similarity_score(y_testset, y_pred_svm)
jacc_svm


# In[104]:


f1_svm= f1_score(y_testset, y_pred_svm , average='weighted')
f1_svm


# In[107]:


y_pred_svm_prob = svm.predict_proba(X_testset)
logl_svm = svm_loss(y_testset, y_pred_svm_prob)
logl_svm


# In[110]:


# predict_proba is not available when  probability=False

logl_svm= 'NA'


# # Report

# In[111]:


Jaccard = [round(x,2) for x in [jacc_knn, jacc_dt, jacc_svm, jacc_lr]] 
F1_Score = [round(x,2) for x in [f1_knn, f1_dt, f1_svm, f1_lr]]
LogLoss = [logl_knn, logl_dt, logl_svm, round(logl_lr,2)]


# In[112]:


z_results = list(zip(Jaccard, F1_Score, LogLoss))


# In[113]:


Report = pd.DataFrame(z_results, columns = ['Jaccard' , 'F1-score', 'LogLoss'], index=['KNN', 'Decision Tree', 'SVM', 'LogisticRegression'])
Report


# In[ ]:




