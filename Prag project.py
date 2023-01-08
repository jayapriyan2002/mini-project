#!/usr/bin/env python
# coding: utf-8

# In[437]:


import os
import warnings
warnings.simplefilter('ignore')


# In[438]:


import numpy as np
import pandas as pd


# In[439]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[440]:


from skimage.io import imread,imshow
from skimage.transform import resize
from skimage.color import rgb2gray


# In[441]:


ian=os.listdir("D:/image/ian/")


# In[442]:


selena=os.listdir("D:/image/selena/")


# In[443]:


erin=os.listdir("D:/image/erin/")


# In[444]:


limit=10

ian_images=[None]*limit
j=0
for i in ian:
    if(j<limit):
        ian_images[j]=imread("D:/image/ian/"+i)
        j+=1
    else:
        break


# In[445]:


limit=10

selena_images=[None]*limit
j=0
for i in selena:
    if(j<limit):
        selena_images[j]=imread("E:/image/selena/"+i)
        j+=1
    else:
        break


# In[446]:


limit=10

erin_images=[None]*limit
j=0
for i in erin:
    if(j<limit):
        erin_images[j]=imread("E:/image/erin/"+i)
        j+=1
    else:
        break


# In[447]:


imshow(ian_images[4])


# In[448]:


imshow(selena_images[7])


# In[449]:


imshow(erin_images[4])


# In[450]:


ian_gray=[None]*limit
j=0

for i in ian:
    if(j<limit):
        ian_gray[j]=rgb2gray(ian_images[j])
        j+=1
    else:
        break


# In[451]:


selena_gray=[None]*limit
j=0

for i in selena:
    if(j<limit):
        selena_gray[j]=rgb2gray(selena_images[j])
        j+=1
    else:
        break


# In[452]:


erin_gray=[None]*limit
j=0

for i in erin:
    if(j<limit):
        erin_gray[j]=rgb2gray(erin_images[j])
        j+=1
    else:
        break


# In[453]:


imshow(ian_gray[4])


# In[454]:


imshow(selena_gray[7])


# In[455]:


imshow(erin_gray[4])


# In[456]:


selena_gray[5].shape


# In[457]:


for j in range (10):
    ia=ian_gray[j]
    ian_gray[j]=resize(ia,(512,512))


# In[458]:


for j in range (10):
    sg=selena_gray[j]
    selena_gray[j]=resize(sg,(512,512))


# In[459]:


for j in range (10):
    er=erin_gray[j]
    erin_gray[j]=resize(er,(512,512))


# In[460]:


imshow(selena_gray[5])


# In[461]:


selena_gray[1].shape
erin_gray[4].shape


# # Ian Somerhalder

# In[462]:


len_of_images_ian=len(ian_gray)


# In[463]:


len_of_images_ian


# In[464]:


image_size_ian=ian_gray[1].shape


# In[465]:


image_size_ian


# In[466]:


flatten_size_ian=image_size_ian[0]*image_size_ian[1]


# In[467]:


flatten_size_ian


# In[468]:


for i in range(len_of_images_ian):
    ian_gray[i]=np.ndarray.flatten(ian_gray[i].reshape(flatten_size_ian,1))


# In[469]:


ian_gray=np.dstack(ian_gray)


# In[470]:


ian_gray.shape


# In[471]:


ian_gray=np.rollaxis(ian_gray,axis=2,start=1)


# In[472]:


ian_gray=ian_gray.reshape(len_of_images_ian,flatten_size_ian)


# In[473]:


ian_gray.shape


# In[474]:


ian_data=pd.DataFrame(ian_gray)


# In[475]:


ian_data


# In[476]:


ian_data["label"]="Ian Somerhalder"


# In[477]:


ian_data


# # Selena Gomez

# In[478]:


len_of_images_selena=len(selena_gray)


# In[479]:


len_of_images_selena


# In[480]:


image_size_selena=selena_gray[1].shape


# In[481]:


image_size_selena


# In[482]:


flatten_size_selena=image_size_selena[0]*image_size_selena[1]


# In[483]:


flatten_size_selena


# In[484]:


for i in range(len_of_images_selena):
    selena_gray[i]=np.ndarray.flatten(selena_gray[i].reshape(flatten_size_selena,1))


# In[485]:


selena_gray=np.dstack(selena_gray)


# In[486]:


selena_gray.shape


# In[487]:


selena_gray=np.rollaxis(selena_gray,axis=2,start=1)


# In[488]:


selena_gray=ian_gray.reshape(len_of_images_ian,flatten_size_ian)


# In[489]:


selena_gray.shape


# In[493]:


selena_data=pd.DataFrame(selena_gray)


# In[494]:


selena_data


# In[495]:


selena_data["label"]="selena gomez"


# In[496]:


selena_data


# # Erin Moriarty

# In[497]:


len_of_images_erin=len(erin_gray)


# In[498]:


len_of_images_erin


# In[499]:


image_size_erin=erin_gray[1].shape


# In[500]:


image_size_erin


# In[501]:


flatten_size_erin=image_size_erin[0]*image_size_erin[1]


# In[502]:


flatten_size_erin


# In[503]:


for i in range(len_of_images_erin):
    erin_gray[i]=np.ndarray.flatten(erin_gray[i].reshape(flatten_size_erin,1))


# In[504]:


erin_gray=np.dstack(erin_gray)


# In[505]:


erin_gray.shape


# In[506]:


erin_gray=np.rollaxis(erin_gray,axis=2,start=1)


# In[507]:


erin_gray=erin_gray.reshape(len_of_images_erin,flatten_size_erin)


# In[508]:


erin_gray.shape


# In[509]:


erin_data=pd.DataFrame(erin_gray)


# In[510]:


erin_data


# In[511]:


erin_data["label"]="erin moriarty"


# In[512]:


erin_data


# In[513]:


friend_1=pd.concat([erin_data,selena_data])


# In[514]:


friend=pd.concat([friend_1,ian_data])


# In[515]:


friend


# In[598]:


from sklearn.utils import shuffle


# In[599]:


boy_indexed=shuffle(friend).reset_index()


# In[600]:


boy_indexed


# In[601]:


boy=boy_indexed.drop(['index'],axis=1)


# In[602]:


boy


# In[603]:


x=boy.values[:,:-1]


# In[604]:


y=boy.values[:,-1]


# In[605]:


x


# In[606]:


y


# In[607]:


from sklearn.model_selection import train_test_split


# In[608]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.215,random_state=0)


# In[609]:


from sklearn import svm


# In[610]:


clf=svm.SVC()
clf.fit(x_train,y_train)


# In[611]:


y_pred=clf.predict(x_test)


# In[612]:


y_pred


# In[613]:


for i in (np.random.randint(0,6,4)):
    predicted_images=(np.reshape(x_test[i],(512,512)).astype(np.float64))
    plt.title('Predicted Label: {0}'. format(y_pred[i]))
    plt.imshow(predicted_images,interpolation='nearest',cmap='gray')
    plt.show()


# # Prediction Accuracy

# In[614]:


from sklearn import metrics


# In[615]:


accuracy=metrics.accuracy_score(y_test,y_pred)


# In[616]:


accuracy


# # Error Analysis of Prediction

# In[617]:


from sklearn.metrics import confusion_matrix


# In[618]:


confusion_matrix(y_test,y_pred)


# # Image Prediction

# # SVM Algorithm

# # Assigning Training and Test Dataset

# # Reshuffle

# # Concatenation

# In[ ]:




