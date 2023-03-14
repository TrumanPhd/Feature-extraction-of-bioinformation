# -*- coding: utf-8 -*-
"""
Hyper Automatic feature Optimazition

@author: Truman

proposed and programmed by Guohao Wang 
which hasn`t be published yet
"""

import numpy as np

"instruction----------"
#类间特征对比 Hyper acid feature option, in short: HAO
"input----------------"
#input: maxtrix: matrix of feature of the same catergory type:np.array
#       label: the label of the training datasets        type:np.array
#       test:  the test_datasets(only using for seletion to feed into the model,
#              based on the index learnt from the trainning dataset ) 
#                                                        type:np.array
#       rate:  the top xx% of the feature(for so many we need)
#还可优化:对同类feature输出权重
"output---------------"
#       selected: selected feature of the train_datasets
#       test_s  : selected feature of the test_datasets(from the index of the train)
def HAO(matrix,label,test,rate = 0.2):
    (seqD,feaD) = matrix.shape
    fea_score = []
    for row in range(feaD):
        sum_pos = 0
        sum_neg = 0
        num_pos = 0
        num_neg = 0
        for line in range(seqD):
            if(label[line] == 1):
                sum_pos += matrix[line,row]
                num_pos +=1
            else:
                sum_neg += matrix[line,row]
                num_neg +=1 
        score =  sum_pos/num_pos-sum_neg/num_neg
        fea_score.append(np.maximum(score,-score))   
    indices = np.argsort(fea_score)[-round(feaD*rate):] 
    first = indices[0]
    matrix = matrix.T
    selected = matrix[first:first+1]
    for i in indices[1:]:
        selected = np.concatenate((selected,matrix[i:i+1]),axis = 0)
    selected = selected.T
    test = test.T
    test_s = test[first:first+1]
    for j in indices[1:]:
        test_s = np.concatenate((test_s,test[j:j+1]),axis = 0)
    test_s = test_s.T
    
    return selected,test_s

#函数介绍本函数根据测试集的数据和标签筛选出特征并返回到selected，为了方便测试，自动根据索引返回测试集的特征矩阵
