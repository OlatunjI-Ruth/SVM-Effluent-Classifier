#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# In[ ]:





# In[2]:


d1 = pd.read_csv('nonscaled1_varibles.csv',  index_col = 0)
d1.insert(7, 'Target', 0)
d1


# In[3]:


d2 = pd.read_csv('nonscaled2_varibles.csv',  index_col = 0)
#d2.drop(columns = ['effluent_turbidity', 'ETSS', 'E_CHLO'], inplace = True)
d2.insert(7, 'Target', 1)
d2


# In[4]:



d2.rename(columns = {'effluent_Ph':'PH', 'Effluent_COD':'COD', 'Effluent_Bod':'BOD', 'Effluent_DO':'DO', 'Effluent_Nitrogen':'NITROGEN', 'Efffluent_phos':'PHOSPHOROUS', 'ETDS':'TDS'}, inplace = True)
d2


# In[5]:


d1.rename(columns = {'Influent_pH':'PH', 'Influent_COD':'COD', 'Influent_BOD':'BOD', 'Influent_DO':'DO', 'Influent_Nitrogen':'NITROGEN', 'Influent_Phosphorous':'PHOSPHOROUS', 'Influent_TDS':'TDS'}, inplace = True)
d1


# In[ ]:





# In[6]:


frames = [d1, d2]
#d1.append(d2)
#ready_data = pd.concat(frames, axis = 1)


# In[7]:


new_frame = pd.concat(frames, axis = 0)


# In[8]:


from sklearn.utils import shuffle
new_frame = shuffle(new_frame)


# In[9]:


#new_frame = new_frame.sample(frac = 1)
new_frame
new_frame.head()


# In[10]:


new_frame.describe()


# In[11]:


#new_frame.loc[new_frame['PH'] > 6.5 | new_frame['PH'] < 8.0 : new_frame['Target'] == safe]

#Target = new_frame[new_frame['PH'] > 6.5 | new_frame['PH'] < 8.0 == 1.0]

conditions = [(6.5 >= (new_frame['PH'] <= 9.0)) & (new_frame['COD'] <= 90) & (new_frame['BOD'] <= 50) & (new_frame['DO'] <= 10) & (new_frame['NITROGEN'] <= 10) & (new_frame['PHOSPHOROUS'] <= 5) & (new_frame['TDS'] <= 500)]

value = ['safe']

new_frame['Target'] = np.select(conditions, value)


new_frame['Target'] = new_frame['Target'].replace(['0'], 'Not safe')


# In[12]:


new_frame


# In[13]:


new_frame['Target'].value_counts()


# In[14]:


test = [12.07999824, 525.00012093,  62.99999876,   4.58057252, 2.1999998 ,   1.30000006, 510.99992726]


# In[15]:


7.77000018, 38.00021553, 17.50000525,  4.29999993,  2.2999997, 1.20000009, 12.99989542


# In[16]:


from sklearn.preprocessing import StandardScaler


# In[17]:


#scaler = StandardScaler()


# In[18]:


#scaled_data = st_x.fit_transform(X_data)


# In[19]:


#scaled_features = scaler.fit_transform(new_frame.drop('Target',axis=1))
#scaled_features


# In[20]:


#independent = pd.DataFrame(scaled_features,columns=new_frame.columns[: -1])
#independent.loc[229]


# In[21]:


independent = new_frame.drop('Target',axis=1)


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(independent,new_frame['Target'],
                                                   test_size=0.30, random_state = 42)


# In[24]:


x_train


# In[25]:


x_test.head(60)


# In[26]:


from sklearn.svm import SVC


# In[27]:


classifier = SVC(kernel = 'rbf', random_state = 42)
classifier.fit(x_train, y_train)


# In[28]:


y_pred = classifier.predict(x_test)


# In[29]:


test_set = [[11.88, 420.0, 63.0, 3.8, 2.90, 1.70, 405.0]]


# In[30]:


y_pred1 = classifier.predict(test_set)
y_pred1


# In[32]:


test_set2 = [[6.60, 60.0, 20.5, 20.0, 1.50, 2.00, 250.0]]


# In[33]:


y_pred2 = classifier.predict(test_set2)
y_pred2


# In[36]:


test_set3 = [[12, 250.0, 20, 7.0, 9.0, 4.0, 500.0]]


# In[38]:


y_pred = classifier.predict(test_set3)
y_pred


# In[37]:


test_set4 = [[6.78, 50.0, 20, 7.0, 9.0, 4.0, 50.0]]


# In[39]:


y_pred3 = classifier.predict(test_set4)
y_pred3


# In[43]:


#performance evaluation
from sklearn.metrics import classification_report,confusion_matrix
pd.DataFrame(confusion_matrix(y_test, y_pred),
             index=['Actual: Unsafe', 'Actual: Safe'],
             columns = ['pred: Unsafe', 'pred: Safe'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[31]:


#y_pred1 = scaler.inverse_transform(test_set)
#y_pred1


# In[34]:


#from sklearn.decomposition import PCA
#pca = PCA(n_components = 2).fit(x_train)
#pca_2d = pca.transform(x_train)


# In[35]:


#pca_2d


# In[ ]:





# In[ ]:





# In[40]:


def values_in(y_pred):
    if conditions == True:
        return 'safe'
    else:
        return 'unsafe'


# In[ ]:


y_test.value_counts()


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:




