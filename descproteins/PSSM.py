#!/usr/bin/env python
#_*_coding:utf-8_*_

import sys, os, platform
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
father_path = os.path.abspath(
    os.path.dirname(pPath) + os.path.sep + ".") + r'\pubscripts' if platform.system() == 'Windows' else os.path.abspath(
    os.path.dirname(pPath) + os.path.sep + ".") + r'/pubscripts'
sys.path.append(father_path)
import check_sequences

def PSSM(fastas, **kw):
    if check_sequences.check_fasta_with_equal_length(fastas) == False:
        print('Error: for "PSSM" encoding, the input fasta sequences should be with equal length. \n\n')
        return 0

    pssmDir = kw['path']
    if pssmDir == None:
        print('Error: please specify the directory of predicted protein disorder files by "--path" \n\n')
        return 0

    AA = 'ARNDCQEGHILKMFPSTWYV'

    encodings = []
    header = ['#', 'label']
    for p in range(1, len(fastas[0][1]) + 1):
        for aa in AA:
            header.append('Pos.'+str(p) + '.' + aa)
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], i[1], i[2]
        code = [name, label]
        if os.path.exists(pssmDir+'/'+name+'.pssm') == False:
            print('Error: pssm profile for protein ' + name + ' does not exist.')
            sys.exit(1)
        with open(pssmDir+'/'+name+'.pssm') as f:
            records = f.readlines()[3: -6]

        proteinSeq = ''
        pssmMatrix = []
        for line in records:
            array = line.strip().split()
            pssmMatrix.append(array[2:22])
            proteinSeq = proteinSeq + array[1]

        pos = proteinSeq.find(sequence)
        if pos == -1:
            print('Warning: could not find the peptide in proteins.\n\n')
        else:
            for p in range(pos, pos + len(sequence)):
                code = code + pssmMatrix[p]
        encodings.append(code)

    return encodings