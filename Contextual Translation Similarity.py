# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import spatial
from wikipedia2vec import Wikipedia2Vec

def merge_ST_TT(df):
    output = {}
    for i in range(len(df)):
        Text = df['Text'][i]
        Id = df['Id'][i]
        SToken = df['SToken'][i]
        TGroup = str(df['TGroup'][i])
        
        vec = 0
        for item in TGroup.split('_'):
            try:
                vec = vec + wiki2vec.get_word_vector(item)
            except Exception as e:
                vec = ''
                break

        if type(vec) != str:
            try:
                if type(output[(Text, Id, SToken)][0]) != str:
                    output[(Text, Id, SToken)].append(vec)
            except Exception as e:
                output[(Text, Id, SToken)] = [vec]
        else:
            output[(Text, Id, SToken)] = ['']
    return output

def contextual_translation_similarity(ST_TT):
    output = []
    for item in ST_TT:       
        if type(ST_TT[item][0]) != str: 
            if len(ST_TT[item]) != 1:            
                euc_cum, cos_cum = 0, 0
                for x in ST_TT[item]:
                    for y in ST_TT[item]:
                        euc_cum = euc_cum + np.linalg.norm(x - y)
                        cos_cum = cos_cum + spatial.distance.cosine(x, y)
                euc = -euc_cum / sum(range(len(ST_TT[item])))
                cos = 1 - cos_cum / (2 * sum(range(len(ST_TT[item]))))
                
                c = 0
                for vec in ST_TT[item]:
                    c = c + vec                   
                c = c / len(ST_TT[item])
                mv_cum = 0
                for x in ST_TT[item]:
                    x = x - c
                    x = x.reshape((len(x), 1))
                    tp_x = np.transpose(x)
                    mv_cum = mv_cum + float(np.dot(tp_x, x))
                mv = mv_cum / len(ST_TT[item])
                
            else:
                euc = 0
                cos = 1
                mv = 0               
        else:
            euc = ''
            cos = ''
            mv = ''
        output.append([item[0], item[1], item[2], euc, cos, mv])  
    return pd.DataFrame(output, columns=['Text', 'Id', 'SToken', 'CTS_EUC', 'CTS_COS', 'CTS_MV'])

def add_entropy(ST_table, CTS, feature):
    output = []
    for i in range(len(CTS)):        
        Text = CTS['Text'][i]
        Id = CTS['Id'][i]
        for item in ST_table[(ST_table['Text']==Text) & (ST_table['Id']==Id)][feature]:
            output.append(item)
            break
    CTS[feature] = output
    return CTS

def add_mean_dur(ST_table, CTS):
    output = []
    for i in range(len(CTS)):        
        Text = CTS['Text'][i]
        Id = CTS['Id'][i]

        cum = 0
        for item in ST_table[(ST_table['Text']==Text) & (ST_table['Id']==Id)]['Dur']:
            cum = cum + item
        mean = cum / len(ST_table[(ST_table['Text']==Text) & (ST_table['Id']==Id)]['Dur'])
        
        output.append(mean)
    CTS['mDur'] = output
    return CTS

def append_CTS(ST_table, CTS):
    euc, cos, mv = [], [], []
    for i in range(len(ST_table)):
        Text = ST_table['Text'][i]
        Id = ST_table['Id'][i]
        for item in CTS[(CTS['Text']==Text) & (CTS['Id']==Id)]['CTS_EUC']:
            euc.append(item)
        for item in CTS[(CTS['Text']==Text) & (CTS['Id']==Id)]['CTS_COS']:
            cos.append(item)
        for item in CTS[(CTS['Text']==Text) & (CTS['Id']==Id)]['CTS_MV']:
            mv.append(item)
            
    ST_table['CTS_EUC'] = euc
    ST_table['CTS_COS'] = cos
    ST_table['CTS_MV'] = mv   
    return ST_table

