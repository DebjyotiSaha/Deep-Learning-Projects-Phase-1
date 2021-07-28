#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


# In[2]:


model=ResNet50(weights='imagenet')


# In[4]:


img_path=r'C:\Users\Hp\Pictures\pug.jpg'


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


from skimage.io import imread


# In[9]:


pip install scikit-image


# In[11]:


img=imread(img_path)


# In[13]:


plt.imshow(img)


# In[14]:


img=image.load_img(img_path, target_size=(224, 224))
x= image.img_to_array(img)
x=np.expand_dims(x, axis=0)
x=preprocess_input(x)

preds=model.predict(x)


# In[15]:


preds


# In[16]:


print('Predicted:', decode_predictions(preds, top=3)[0])


# In[ ]:




