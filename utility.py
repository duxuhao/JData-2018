import numpy as np
import pandas as pd

def readlog(fn,s):
    with open(fn,'r') as f:
        a = f.readline().strip()
        if len(a) < 2:
            a = a + ' '
        while (a[0] != '*') | (a[1:] != str(s)):
            a = f.readline().strip()
            if len(a) < 2:
                a = a + ' '
        a = f.readline().strip().split(' ')
    return a

def score(pred, real):
    pred['index'] = np.arange(pred.shape[0]) + 1
    pred['wi'] = 1 / (1 + np.log(pred['index']))

    compare = pd.merge(pred, real, how='left', on='user_id')
    compare = compare[:50000]
    compare.fillna(0, inplace=True)
    S1 = np.sum(compare['buy'] * compare['wi']) / compare['wi'].sum()

    compare_for_S2 = compare[compare['buy'] == 1]
    S2 = np.sum(10 / (10 + np.square(compare_for_S2['Days'] - compare_for_S2['nextbuy']))) / real.shape[0]

    S = 0.4 * S1 + 0.6 * S2
    print("S1=", S1, "| S2 ", S2)
    print("S =", S)
    return S

def score1(pred, real):
    pred['index'] = np.arange(pred.shape[0]) + 1
    pred['wi'] = 1 / (1 + np.log(pred['index']))

    compare = pd.merge(pred, real, how='left', on='user_id')
    compare = compare[:50000]
    compare.fillna(0, inplace=True)
    S1 = np.sum(compare['buy'] * compare['wi']) / compare['wi'].sum()
    return S1

def Merge(fnm,fn,key):
    df = pd.read_csv(fnm)
    mergecol = ['user_id','CreateGroup']
    col = []
    for k in key:
        col = col + [i for i in df.columns if k in i]
    for i in fn:
        temp = pd.read_csv(i)
        post = i.split('.')[0].split('_')[1]
        coltemp = ["{}_{}".format(j, post) for j in col]
        temp = temp[mergecol + col]
        temp.columns = mergecol + coltemp
        df = df.merge(temp, on = mergecol, how = 'left')
    return df

def Merge2(fnm,fn,key):
    df = pd.read_csv(fnm)
    mergecol = ['user_id','CreateGroup']
    col = []
    for i in fn:
        temp = pd.read_csv(i)  
        for k in key:
            col = col + [i for i in temp.columns if k in i]
        post = i.split('.')[0].split('_')[1]
        coltemp = ["{}_{}".format(j, post) for j in col]
        temp = temp[mergecol + col]
        temp.columns = mergecol + coltemp
        df = df.merge(temp, on = mergecol, how = 'left')
    return df
