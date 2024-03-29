#!/usr/bin/env python
#_*_coding:utf-8_*_

import re
from collections import Counter

def AAC(fastas, **kw):
    AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
    #AA = 'ARNDCQEGHILKMFPSTWYV'
    encodings = []
    header = ['#', 'label']
    for i in AA:
        header.append(i)
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        count = Counter(sequence)
        for key in count:
            count[key] = count[key]/len(sequence)
        code = [name, label]
        for aa in AA:
            code.append(count[aa])
        encodings.append(code)
    return encodings