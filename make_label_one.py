#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from skimage import io


# In[2]:


print('input image data directory name (please include last /)')
image_dir = input()
print('input annotation data path (please include .csv)')
annotation_data_path = input()
annotatin_data = pd.read_csv(annotation_data_path)
print('input output data directory name (please include last /)')
output_dir = input()


# In[3]:


λ = 1.0
# 画像の置いてあるディレクトリ名 image_dir、画像のファイル名 image_name、アノテーションデータのcsvファイル annotation_csv、
# バウンディングボックスから標準偏差を抽出するときに用いるkシグマ区間のkを表す整数 k_sigma、を引数に取り、
# ラベルを生成してnp.array形式で返す関数
def make_label_one(image_dir, image_name, annotation_csv):
    image_shape = io.imread(image_dir + image_name).shape[:2]
    label_np = np.zeros(image_shape)

    image_name, _ = image_name.split('.')
    annotation_csv_for_image = annotation_csv[annotation_csv.Image == image_name]
    
    if len(annotation_csv_for_image) == 0:
        return label_np
    
    annotation_data_bounding_box = annotation_csv_for_image[['X', 'Y', 'Width', 'Height']].values.tolist()
    
    for X, Y, Width, Height in annotation_data_bounding_box:
        Height = round(Height*λ)
        Width = round(Width*λ)
        for i in range(Y, Y+Height):
            for j in range(X, X+Width):
                label_np[i][j] = 1
                        
    return label_np


# In[ ]:


if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
files = os.listdir(image_dir)


# In[5]:


for f in files:
    label = make_label_one(image_dir, f, annotation_data)
    io.imsave(output_dir + f, label)


# In[ ]:




