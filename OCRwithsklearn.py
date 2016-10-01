# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 11:08:00 2016

@author: 10150
"""
from __future__ import division

import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics

def precision(expected,predicted):
    n_wrong=0
    for i,j in zip(range(len(metrics.confusion_matrix(expected,predicted)[0])),
                   range(len(metrics.confusion_matrix(expected,predicted)[1]))):
        if i==j:
            n_wrong+=metrics.confusion_matrix(expected,predicted)[i,j]
    print "预测精度为:%.2f%%" %(n_wrong/len(expected)*100)

digits=datasets.load_digits()
images_and_labels=list(zip(digits.images,digits.target))

#将图形数据转换为向量
n_sample=len(digits.images)
data=digits.images.reshape(n_sample,-1)
    
#构造支持向量机模型
svm_classifier=svm.SVC(gamma=0.001,kernel='poly')
#训练模型，用前半部分数据
svm_classifier.fit(data[:int(n_sample/2)],digits.target[:int(n_sample/2)])

expected_labels=digits.target[int(n_sample/2):]
#预测
predicted_labels=svm_classifier.predict(data[int(n_sample/2):])

print "Classification report for classifier: %s\n%s" %(svm_classifier,
            metrics.classification_report(expected_labels,predicted_labels))
print "Confusion matrix:\n%s" %metrics.confusion_matrix(expected_labels,predicted_labels)

#显示测试数据和预测标签
images_and_prediction=list(zip(digits.images[int(n_sample/2):],predicted_labels))

'''
#绘制原始数据的图像
plt.figure()
for index,(image,label) in enumerate(images_and_labels[int(n_sample/2):int(n_sample/2)+12]):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='bessel')
    plt.title('Real: %d' %label)
    
#绘制预测图像的图形
plt.figure()
for index,(image,label) in enumerate(images_and_prediction[:12]):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='bessel')
    plt.title('Prediction:%d'%label)
'''
#计算预测的准确率
n_right=0
for i,j in zip(range(len(metrics.confusion_matrix(expected_labels,predicted_labels)[0])),
               range(len(metrics.confusion_matrix(expected_labels,predicted_labels)[1]))):
    if i==j:
        n_right+=metrics.confusion_matrix(expected_labels,predicted_labels)[i,j]
print "预测精度为:%.2f%%" %(n_right/len(expected_labels)*100)
  
#绘制预测错误的图像
ima_ex_pred=list(zip(digits.images[int(n_sample/2):],
                                   digits.target[int(n_sample/2):],
                                                 predicted_labels))
plt.figure()
index_wrong=0
for index,(image,label_ex,label_pred) in enumerate(ima_ex_pred):
    if label_ex!=label_pred:
        index_wrong+=1
        plt.subplot(10,5,index_wrong)
        plt.axis('off')
        plt.imshow(image,cmap=plt.cm.gray_r,interpolation='bessel')
        plt.title('r:{0},p:{1}'.format(label_ex,label_pred))
print "预测错的个数为：%d" %index_wrong