# -*- coding: utf-8 -*-
"""
HAO_tool a tool for bioinformation 
 
@author: Truman
part 1:
    --protein
    --classic feature
    this is a part of feature extraction 
    it has cover all the protein feature extraction approach 
"""

import numpy as np
import pandas as pd
import sklearn 
import PseudoAAC

"------------------------PseAAC encoding----------------------"
#input: seq of 2220
#output: a numpyarray (20 + lamda)D lamda = 40
#has been definated in the file PseudoAAC
def PseAAC(loc):
    Pse = PseudoAAC.PseAAC(loc)
    while len(Pse) != 60:
        Pse.extend('0')
    Pse = list(map(int,Pse))
    return Pse

"------------------------sumaa-------------------------"
#input: seqs
#output: 20D array of the aa number of 20kinds aa 
def sumaa(seqs):
    length = len(seqs)
    #create a list；
    AAnum = [0]*20
    AA=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    SUMA = dict(zip(AA,AAnum))
    for aa in seqs:
        for a in SUMA.keys():
            if aa == a:
                SUMA[a] += 1
                break
    res = []
    for i in list(SUMA.values()):
        res.append(i/length)
    return np.array(res)
"---------------------------kmer-------------------------"
def kmer(seq):
    mer2={}
    mer3={}
    for n1 in 'ARNDCEQGHILKMFPSTWYV':
        for n2 in 'ARNDCEQGHILKMFPSTWYV':
            mer2[n1+n2]=0
            for n3 in 'ARNDCEQGHILKMFPSTWYV':
                mer3[n1+n2+n3]=0
    seq_len=len(seq)
    for p in range(0,seq_len-2):
        mer2[seq[p:p+2]]+=1
        mer3[seq[p:p+3]]+=1
    mer2[seq[p+1:p+3]]+=1
    v2=[]
    v3=[]
    for n1 in 'ARNDCEQGHILKMFPSTWYV':
        for n2 in 'ARNDCEQGHILKMFPSTWYV':
            v2.append(mer2[n1+n2])
            for n3 in 'ARNDCEQGHILKMFPSTWYV':
                v3.append(mer3[n1+n2+n3])
    v=v2+v3 
    return np.array(v)

"----------------------------FE------------------------------"
def FE(seq):
    len_seq=len(seq)
    n={}#平均数
    u={}#期望
    D={}#方差
    f={}#例如，第i个位置时AA的数量
    v=[]
    for n1 in 'ARNDCEQGHILKMFPSTWYV':
        for n2 in 'ARNDCEQGHILKMFPSTWYV':
            n[n1+n2]=0
            f[n1+n2]=0
            u[n1+n2]=0
            D[n1+n2]=0
    for i in range(0,len_seq-1):###采用累加累乘计算二元核苷酸的平均次数、期望及其方差
        n[seq[i:i+2]]+=1
        f[seq[i:i+2]]=1
        u[seq[i:i+2]]+=(i+1)*f[seq[i:i+2]]/float(n[seq[i:i+2]])
        t=(i+1-u[seq[i:i+2]])*(i+1-u[seq[i:i+2]])
        D[seq[i:i+2]]+=t*f[seq[i:i+2]]/float(n[seq[i:i+2]]*(len_seq-1))
    for n1 in 'ARNDCEQGHILKMFPSTWYV':
        for n2 in 'ARNDCEQGHILKMFPSTWYV':
            v.append(n[n1+n2])
            v.append(u[n1+n2])
            v.append(D[n1+n2])
    return np.array(v)

"""-------------------------DRA-------------------------"""
"""计算二联核苷酸出现概率与两个核苷酸单独出现的概率比，用于衡量相关性"""
###二联核苷酸相对丰度特征提取方法，衡量两个相邻碱基之间的相关性
def DRA(seq):
    mer1={}
    mer2={}
    for n1 in 'ARNDCEQGHILKMFPSTWYV':###给单元、二元核苷酸出现次数赋初值0
        mer1[n1]=0
        for n2 in 'ARNDCEQGHILKMFPSTWYV':
            mer2[n1+n2]=0 #生成但核苷酸和二联和二联核苷酸计数的字典，并赋初值等于0。
    seq_len=len(seq)
    for p in range(0,seq_len-1):###计算单元、二元核苷酸出现次数
        mer1[seq[p:p+1]]+=1
        mer2[seq[p:p+2]]+=1
    mer1[seq[p+1:p+2]]+=1    
    v1={}
    v2={}
    T={}
    for n1 in 'ARNDCEQGHILKMFPSTWYV':###计算单元、二元核苷酸出现频率
        v1[n1]=mer1[n1]/float((seq_len))
        for n2 in 'ARNDCEQGHILKMFPSTWYV':
            v2[n1+n2]=mer2[n1+n2]/float((seq_len-1))
    for n1 in 'ARNDCEQGHILKMFPSTWYV':###计算二联核苷酸相对丰度
        for n2 in 'ARNDCEQGHILKMFPSTWYV':
            if v1[n1]*v1[n2]==0: ###不存在的二元核苷酸指定其相对丰度为0
                T[n1+n2]=0
            else:
                T[n1+n2]=(v2[n1+n2]/(v1[n1]*v1[n2]))
    f=[]  
    for n1 in 'ARNDCEQGHILKMFPSTWYV':
        for n2 in 'ARNDCEQGHILKMFPSTWYV':
            f.append(T[n1+n2])
    return np.array(f) #返回16种二联碱基的相关性矩阵


"""------------------------ksnpf编码-------------------------"""

def ksnpf(seq):
    kn=20
    freq=[]
    v=[]
    for i in range(0,kn):
        freq.append({})
        for n1 in 'ARNDCEQGHILKMFPSTWYV':
            freq[i][n1]={}
            for n2 in 'ARNDCEQGHILKMFPSTWYV':
                freq[i][n1][n2]=0 #生成一个列表，含有kn个字典，每个字典5个键（AGCTN),对应的值为统计AGCTN个数的字典，并赋初值为0。
    seq_len=len(seq)
    for k in range(0,kn):
        for i in range(seq_len-k-1):
            n1=seq[i]
            n2=seq[i+k+1]
            freq[k][n1][n2]+=1#统计个数
    for i in range(0,kn):
        for n1 in 'ARNDCEQGHILKMFPSTWYV':
            for n2 in 'ARNDCEQGHILKMFPSTWYV':
                v.append(freq[i][n1][n2])
    return np.array(v) #将多维字典转化为一个一维的列表并返回




"------------------------------test-----------------------------"
def test():
    return np.array([1,2])

"-----------------------------length----------------------------"
 
def lenth(seqs):
    #get the length 
    length = []
    for seq in seqs:
        length.append(len(seq))
    
    return length    


               

"""
"---------------------------fun1-----------------------------"
def fun1(pssm_file,seqs,extraction_tool):     #A function to get feature
    
    #length = len(seqs)
    #PSSM
    feature_array = PSSM(pssm_file)
    feature_PSSM = feature_array.reshape(820)
    feature = feature_PSSM
    #PKA: input PSSM 
    if 1 in extraction_tool:
        feature_PKA = PKA(feature_array)
        feature = np.append(feature_PSSM,feature_PKA)
    #TCP: input PSSM
    if 2 in extraction_tool:
        feature_TCP = TCP(feature_array)
        feature = np.append(feature,feature_TCP)
    #PseAAC: input location of the sequence
    if 3 in extraction_tool:
        feature_PseAAC = PseAAC(seqs)
        feature = np.append(feature,feature_PseAAC)
    if 4 in extraction_tool:
        feature = np.append(feature,sumaa(seqs))
    if 5 in extraction_tool:   
        feature = np.append(feature,test())
    if 6 in extraction_tool:   
        feature = np.append(feature,kmer(seqs))        
    if 7 in extraction_tool:   
        feature = np.append(feature,FE(seqs))   
    if 8 in extraction_tool:   
        feature = np.append(feature,DRA(seqs))   
    if 9 in extraction_tool:   
        feature = np.append(feature,ksnpf(seqs))   
        
    if 0 in extraction_tool:
        pass
    else:    
        feature = feature[820:]

    return feature
"""
