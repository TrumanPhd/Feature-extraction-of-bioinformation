# -*- coding: utf-8 -*-
"""
Feature extraction for DNA or RNA

@author: Guohao Wang(Truman)
"""

import itertools
import pickle
import argparse
import os,sys,re
import numpy as np
from collections import Counter
from keras.models import load_model
import xgboost as xgb
import pandas as pd
import PseudoAAC

#below here are wrapped fun which can be derictly used
def binary(sequences):
    AA = ''
    binary_feature = []
    for seq in sequences:
        binary = []
        for aa in seq:
            for aa1 in AA:
                tag = 1 if aa == aa1 else 0
                binary.append(tag)
        binary_feature.append(binary)
    return binary_feature

def CKSNAP(sequences):
    K=2
    cksnap_feature = []
    AA = 'ACGU'
    AApairs = []
    for aa1 in AA:
        for aa2 in AA:
            AApairs.append(aa1 + aa2)
    for seq in sequences:
        cksnap = []
        l = len(seq)
        for k in range(0, K + 1):
            record_dict = {}
            for i in AApairs:
                record_dict[i] = 0

            sum = 0
            for index1 in range(l):
                index2 = index1 + k + 1
                if index1 < l and index2 < l:
                    record_dict[seq[index1] + seq[index2]] = record_dict[seq[index1] + seq[index2]] + 1
                    sum = sum + 1

            for pair in AApairs:
                cksnap.append(record_dict[pair] / sum)
        cksnap_feature.append(cksnap)
    return cksnap_feature


def NCP(sequences):
    chemical_property = {
        'A': [1, 1, 1],
        'C': [0, 1, 0],
        'G': [1, 0, 0],
        'U': [0, 0, 1], }
    ncp_feature = []
    for seq in sequences:
        ncp = []
        for aaindex, aa in enumerate(seq):
            ncp = ncp + chemical_property.get(aa, [0, 0, 0])
        ncp_feature.append(ncp)
    return ncp_feature


def ND(sequences):
    nd_feature = []
    for seq in sequences:
        nd = []
        for aaindex, aa in enumerate(seq):
            nd.append(seq[0: aaindex + 1].count(seq[aaindex]) / (aaindex + 1))
        nd_feature.append(nd)
    return nd_feature



def ENAC(sequences):
    AA = 'ACGU'
    enac_feature = []
    window = 2
    for seq in sequences:
        l = len(seq)
        enac= []
        for i in range(0, l):
            if i < l and i + window <= l:
                count = Counter(seq[i:i + window])
                for key in count:
                    count[key] = count[key] / len(seq[i:i + window])
                for aa in AA:
                    enac.append(count[aa])
        enac_feature.append(enac)
    return enac_feature


myDiIndex = {
    'AA': 0, 'AC': 1, 'AG': 2, 'AU': 3,
    'CA': 4, 'CC': 5, 'CG': 6, 'CU': 7,
    'GA': 8, 'GC': 9, 'GG': 10, 'GU': 11,
    'UA': 12, 'UC': 13, 'UG': 14, 'UU': 15
}
baseSymbol = 'ACGU'

def npf(seq):
    binary_dictionary={'A':[1,1,1],'T':[0,1,0],'G':[1,0,0],'C':[0,0,1],'N':[0,0,0]}
    cnt=[]
    for i in seq:
        cnt.append(binary_dictionary[i])
    return reduce(operator.add,cnt)
def ssc(seq):
    pname="D:\VRNA\RNAfold.exe"
    source=seq.replace('\n','')
    source=source+'N'
    seq=seq.replace('\n','')
    p=subprocess.Popen(pname,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    result=p.communicate(input=source)
    res=result[0].decode()[0:]
    length=len(seq)
    ssc={}
    ssc_vec={}
    for n1 in 'ATCG':
        for n2 in '.()':
            for n3 in '.()':
                for n4 in '.()':
                    ssc[n1+n2+n3+n4]=0
    res=res.split('N')
    res_str=res[1]
    res_str=res_str.encode()
    res_len=len(res_str)
    res_str=res_str[2:res_len-12]#from 2
    for p in range(0,length-2):
        ssc[seq[p]+res_str[p:p+3]]+=1
    for n1 in 'ATCG':
        ssc_vec[n1+'...']=ssc[n1+'...']
        ssc_vec[n1+'..(']=ssc[n1+'..(']+ssc[n1+'..)']
        ssc_vec[n1+'.(.']=ssc[n1+'.(.']+ssc[n1+'.).']
        ssc_vec[n1+'(..']=ssc[n1+'(..']+ssc[n1+')..']
        for n2 in '()':
            for n3 in '()':
                ssc_vec[n1+'.((']=ssc[n1+'.'+n2+n3]
        for n2 in '()':
            for n3 in '()':
                ssc_vec[n1+'(.(']=ssc[n1+n2+'.'+n3]
        for n2 in '()':
            for n3 in '()':
                ssc_vec[n1+'((.']=ssc[n1+n2+n3+'.']
        for n2 in '()':
            for n3 in '()':
                for n4 in '()':
                    ssc_vec[n1+'(((']=ssc[n1+n2+n3+n4]
    v=[]
    for n1 in 'ATCG':
        for n2 in '.(':
            for n3 in '.(':
                for n4 in '.(':
                    v.append(ssc_vec[n1+n2+n3+n4])
#    for n1 in 'ATCG':
#        for n2 in '.()':
#            for n3 in '.()':
#                for n4 in '.()':
#                    v.append(ssc[n1+n2+n3+n4])
    return v

#some of my own created extraction tool, may not leading to perfect results
"--------PKA-------这个先不要用，还要改"
def PKA(sequence,Kmax=4):
    #Kmax:the most distant pair taken into consideration     
    Score_matrix1 = []
    Score_matrix2 = []
    Score_matrix  = []
    S = 0 #sum of the s
    s = 0 #score of each pair
    for k in range(Kmax+1):
        for i in range(4):
            for j in range(4):    
                for m in range(len(sequence)-k-1):
                    s = max(min(sequence[m,i],sequence[m+k+1,j]),0)
                    S = S + s 
                S /= (len(sequence)-k-1)    
                Score_matrix1.append(S)    
                S = 0
            Score_matrix2.append(Score_matrix1)
            Score_matrix1 = []
        #symmetrical optimization
        for row in range(4):
            for line in range(4):
                register = Score_matrix2[line][row]
                Score_matrix2[line][row] = 0
                Score_matrix2[row][line] += register 
        Score_matrix.append(Score_matrix2) 
        Score_matrix2 = []
    Score_matrix = np.array(Score_matrix)
    size_ = (Kmax+1)*4*4
    Score_matrix = Score_matrix.reshape(size_) 
    
    return Score_matrix 

"-------------------------TCP encoding------------------------"
"这个先不要用，还要改"
def TCP(sequence):
    Score_matrix1 = []
    Score_matrix2 = []
    Score_matrix  = []
    S = 0 #sum of the s
    s = 0 #score of each pair
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for t in range(len(sequence)-2):
                    s = max(min(sequence[t,i],sequence[t+1,j],sequence[t+2,k]),0)
                    S += s
                S /= 40 #2*omiga = 40 it could be optimized  
                Score_matrix1.append(S)    
                S = 0
            Score_matrix2.extend(Score_matrix1)
            Score_matrix1 = []
        Score_matrix.extend(Score_matrix2) 
        Score_matrix2 = [] 
    Score_matrix = np.array(Score_matrix)
    Score_matrix = Score_matrix.reshape(8000) 
    return Score_matrix 
     
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

#---------------------------------------------------------------------------
#below there are fun which still should be debug
#--------------------------------------------------------------------------- 
def get_kmer_frequency(sequence, kmer):
    frequency = {}
    for pep in [''.join(i) for i in list(itertools.product(baseSymbol, repeat=kmer))]:
        #itertools.product(A,B),返回A，B中元素的笛卡尔积的元祖，product(A,repeat=4)的含义与product(A,A,A,A)的含义相同。
        frequency[pep] = 0
    for i in range(len(sequence) - kmer + 1):
        frequency[sequence[i: i + kmer]] = frequency[sequence[i: i + kmer]] + 1
    for key in frequency:
        frequency[key] = frequency[key] / (len(sequence) - kmer + 1)
    return frequency #返回的是一个字典‘AAAAA’：0.34




def correlationFunction_type2(pepA, pepB, myIndex, myPropertyName, myPropertyValue):
    CC = 0
    for p in myPropertyName:
        CC = CC + float(myPropertyValue[p][myIndex[pepA]]) * float(myPropertyValue[p][myIndex[pepB]])
    return CC




def get_theta_array_type2(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence,i):
    fixkmer = i
    thetaArray = []
    for tmpLamada in range(lamadaValue):
        for p in myPropertyName:
            theta = 0
            for i in range(len(sequence) - tmpLamada - fixkmer):
                theta = theta + correlationFunction_type2(sequence[i:i + fixkmer],
                                                          sequence[i + tmpLamada + 1: i + tmpLamada + 1 + fixkmer],
                                                          myIndex,
                                                          [p], myPropertyValue)
            thetaArray.append(theta / (len(sequence) - tmpLamada - fixkmer))
    return thetaArray




def SCPseDNC(sequences):
    property_name = ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']
    dataFile = 'dirnaPhyche.data'
    with open(dataFile,'rb') as f:
        property_value = pickle.load(f)
    lamada =  20#20
    weight =  0.9#0.9
    kmer_index = myDiIndex
    SCPseDNC_feature = []
    for i in sequences:
        code = []
        dipeptideFrequency = get_kmer_frequency(i, 2)
        thetaArray = get_theta_array_type2(kmer_index, property_name, property_value, lamada, i,2)
        for pep in sorted(kmer_index.keys()):
            code.append(dipeptideFrequency[pep] / (1 + weight * sum(thetaArray)))
        for k in range(17, 16 + lamada * len(property_name) + 1):
            code.append((weight * thetaArray[k - 17]) / (1 + weight * sum(thetaArray)))
        SCPseDNC_feature.append(code)
    return SCPseDNC_feature