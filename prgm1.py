
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import svm
np.set_printoptions(threshold=np.nan)


# In[2]:



ob = open('train.txt','r')

class_data = np.array([])
abstract_data = np.array([])


# In[3]:


abstract_data


# In[4]:


n = 1
while True:
    line = ob.readline()
    if line == '':
        break
    class_labels = line[0]
    abstract = line[1:].strip()
    class_data = np.append(class_data,class_labels)
    abstract_data = np.append(abstract_data,abstract)
    n = 2
    
#print(class_data)
#print(abstract_data)
print('v')


# In[5]:


len(class_data)


# In[6]:


len(abstract_data)


# In[7]:


clfrSVM = svm.SVC(kernel = 'linear', C= 0.1)


# In[9]:


#clfrSVM.fit(abstract_data,class_data )


# In[10]:


from sklearn.feature_extraction.text import CountVectorizer


# In[11]:


vect = CountVectorizer().fit(abstract_data)


# In[17]:


vect.get_feature_names()[1]


# In[18]:


len(vect.get_feature_names())


# In[19]:


X_Abstract_vectorized = vect.transform(abstract_data)
X_Abstract_vectorized


# In[20]:


clfrSVM.fit(X_Abstract_vectorized,class_data)


# In[23]:


ob_test = open('test.txt')
test = []
while True:
    line = ob_test.readline()
    if line == '':
        break
    test.append(line)

len(test)
    


# In[24]:


predicted_labels = clfrSVM.predict(vect.transform(test))


# In[31]:


len(predicted_labels)
predicted_labels[0]


# In[43]:


f = open('output.txt','w+')


# In[46]:


for i in range(len(predicted_labels)):
    f.write(predicted_labels[i])
    f.write('\n')


# In[47]:


# for i in range(10):
#       f.write("This is line %d\r\n" % (i+1))


# In[48]:


f.close()

