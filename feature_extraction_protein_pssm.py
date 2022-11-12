# -*- coding: utf-8 -*-
"""
@author: Truman
HAO_tool a tool for bioinformation 
 
@author: Truman
part 1:
    --protein
    --feature extraction based on pssm
    this is a part of feature extraction 
    it has cover all the protein feature extraction approach 

"""
import re
import numpy as np
"---------------------PSSM encoding loader--------------------------"
#input Position weight score matrix
def PSSM(pssm_file):
    pssm=[]
    for line in pssm_file:
        line1=re.findall(r"[\-|0-9]+",line)
        del line1[41:]
        pssm.append(line1)
    del pssm[0:3]
    del pssm[-6:]
    
    pssm_array_origin = np.array(pssm,dtype=np.float32)
    #The PSSM matrix is obtained
    pssm_array=pssm_array_origin[:,1:21] 
    
    #cover the void with 0
    while pssm_array.shape != (41,20):
        pssm_array = np.concatenate([pssm_array,np.zeros((1,20))],axis=0)
       
    return pssm_array

"-------------------------PKA encoding------------------------"
#input: PSSM.path
#output: a numpyarray 20x(4+1)x20D 
def PKA(pssm):
    Kmax = 4 #the most distant pair taken into consideration 
    Score_matrix1 = []
    Score_matrix2 = []
    Score_matrix  = []
    S = 0 #sum of the s
    s = 0 #score of each pair
    for k in range(Kmax+1):
        for i in range(20):
            for j in range(20):    
                for m in range(41-k-1):
                    s = max(min(pssm[m,i],pssm[m+k+1,j]),0)
                    S = S + s 
                S /= (41-k-1)    
                Score_matrix1.append(S)    
                S = 0
            Score_matrix2.append(Score_matrix1)
            Score_matrix1 = []
        #symmetrical optimization
        for row in range(20):
            for line in range(20):
                register = Score_matrix2[line][row]
                Score_matrix2[line][row] = 0
                Score_matrix2[row][line] += register 
        Score_matrix.append(Score_matrix2) 
        Score_matrix2 = []
    Score_matrix = np.array(Score_matrix)
    Score_matrix = Score_matrix.reshape(2000) 
    
    return Score_matrix 

"-------------------------TCP encoding------------------------"
#input: PSSM.path
#output: a numpyarray 20x20x20D 
def TCP(pssm):
    Score_matrix1 = []
    Score_matrix2 = []
    Score_matrix  = []
    S = 0 #sum of the s
    s = 0 #score of each pair
    for i in range(20):
        for j in range(20):
            for k in range(20):
                for t in range(41-2):
                    s = max(min(pssm[t,i],pssm[t+1,j],pssm[t+2,k]),0)
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
     
def preHandleColumns(PSSM,STEP,PART,ID):   #A function used in extracting DPC,EEDP and KSB
    '''
    if STEP=k, we calculate the relation betweem one residue and the kth residue afterward.
    '''
    '''
    if PART=0, we calculate the left part of PSSM.
    if PART=1, we calculate the right part of PSSM.
    '''
    '''
    if ID=0, we product the residue-pair.
    if ID=1, we minus the residue-pair.
    '''
    '''
    if KEY=1, we divide each element by the sum of elements in its column.
    if KEY=0, we don't perform the above process.
    '''
    if PART==0:
        PSSM=PSSM[:,1:21]
    elif PART==1:
        PSSM=PSSM[:, 21:]
    PSSM=PSSM.astype(float)
    matrix_final = [ [0.0] * 20 ] * 20
    matrix_final=np.array(matrix_final)
    seq_cn=np.shape(PSSM)[0]

    if ID==0:
        for i in range(20):
            for j in range(20):
                for k in range(seq_cn - STEP):
                    matrix_final[i][j]+=(PSSM[k][i]*PSSM[k+STEP][j])

    elif ID==1:
        for i in range(20):
            for j in range(20):
                for k in range(seq_cn - STEP):
                    matrix_final[i][j] += ((PSSM[k][i]-PSSM[k+STEP][j]) * (PSSM[k][i]-PSSM[k+STEP][j])/4.0)
    return matrix_final


def average(matrixSum, seqLen):            #A function used in extracting DPC,EEDP and KSB
    # average the summary of rows
    matrix_array = np.array(matrixSum)
    matrix_array = np.divide(matrix_array, seqLen)
    matrix_array_shp = np.shape(matrix_array)
    matrix_average = [(np.reshape(matrix_array, (matrix_array_shp[0] * matrix_array_shp[1], )))]
    return matrix_average


def dpc_pssm(input_matrix):   #A function to get DPC
    PART = 0
    STEP = 1
    ID = 0
    KEY = 0
    matrix_final = preHandleColumns(input_matrix, STEP, PART, ID)
    seq_cn = float(np.shape(input_matrix)[0])
    dpc_pssm_vector = average(matrix_final, seq_cn-STEP)
    return dpc_pssm_vector[0]



def eedp(input_matrix):   #A function to get EEDP
    STEP = 2
    PART = 0
    ID = 1
    KEY = 0
    seq_cn = float(np.shape(input_matrix)[0])
    matrix_final = preHandleColumns(input_matrix, STEP, PART, ID)
    eedp_vector = average(matrix_final, seq_cn-STEP)
    return eedp_vector[0]



def k_separated_bigrams_pssm(input_matrix): #A function to get KSB
    PART=1
    ID=0
    KEY=0
    STEP = 1
    matrix_final = preHandleColumns(input_matrix, STEP, PART, ID)
    seq_cn = float(np.shape(input_matrix)[0])
    k_separated_bigrams_pssm_vector=average(matrix_final,10000.0)
    return k_separated_bigrams_pssm_vector[0]
