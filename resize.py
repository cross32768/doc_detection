#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
from PIL import Image


# In[4]:


print('input image data directory name (please include last /)')
image_dir = input()
print('input label data directory name (please include last /)')
label_dir = input()
print('input output data directory name for image (please include last /)')
output_image_dir = input()
print('input output data directory name for label (please include last /)')
output_label_dir = input()


# In[8]:


if not os.path.exists(output_image_dir):
    os.mkdir(output_image_dir)
if not os.path.exists(output_label_dir):
    os.mkdir(output_label_dir)


# In[9]:


image_list = os.listdir(image_dir)


# In[10]:


for image in image_list:
    img = Image.open(image_dir + image)
    img_resize = img.resize((512 ,512), Image.LANCZOS)
    img_resize.save(output_image_dir + image)


# In[ ]:


label_list = os.listdir(label_dir)


# In[12]:


for label in label_list:
    img = Image.open(label_dir + label)
    img_resize = img.resize((512, 512),Image.LANCZOS)
    img_resize.save(output_label_dir + label)

