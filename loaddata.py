# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 11:08:00 2016

@author: 10150
"""

import matplotlib.pyplot as plt
from sklearn import datasets

digits=datasets.load_digits()
images_and_labels=list(zip(digits.images,digits.target))

for index,(image,label) in enumerate(images_and_labels[:10]):
    plt.subplot(2,5,index+1)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Training: %i' %label)