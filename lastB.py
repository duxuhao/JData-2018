import datetime
import numpy as np
import pandas as pd

order = pd.read_csv('./B/jdata_user_order.csv', parse_dates=['o_date'])
sku = pd.read_csv('./B/jdata_sku_basic_info.csv', )
order = pd.merge(order, sku, on='sku_id', how='left')
target_order = order[(order.cate == 101) | (order.cate == 30)].reset_index(drop=True)
first_day = datetime.datetime.strptime('2016-08-31 00:00:00', '%Y-%m-%d %H:%M:%S')
target_order['o_day_series'] = (target_order['o_date'] - first_day).apply(lambda x: x.days)

df2 = pd.read_csv('trainb_1month_final.csv')
dfp = pd.read_csv('testb_1month_final.csv')
f = ['user_id','CreateGroup','last3','last8','last5']
dfp.CreateGroup = 366
df = pd.concat([df2,dfp]).reset_index(drop = True)
#df = df[df.buy == 1].reset_index(drop = True)
temp = df[['user_id','buy','CreateGroup']][:]
for i in range(1,9):
    temp['CreateGroup'] = temp['CreateGroup'] + 30
    temp.columns = ['user_id','buy_{}'.format(i),'CreateGroup']
    df = df.merge(temp,on=['user_id','CreateGroup'],how = 'left')
    f.append('buy_{}'.format(i))

temp = df[['user_id','nextbuy','CreateGroup']][:]
for i in range(1,9):
    temp['CreateGroup'] = temp['CreateGroup'] + 30
    temp.columns = ['user_id','nextbuy_{}'.format(i),'CreateGroup']
    df = df.merge(temp,on=['user_id','CreateGroup'],how = 'left')
    f.append('nextbuy_{}'.format(i))

temp = df[['user_id','buycnt','CreateGroup']][:]
temp.CreateGroup -=  30
for i in range(1,9):
    temp['CreateGroup'] = temp['CreateGroup'] + 30
    temp.columns = ['user_id','buycnt_{}'.format(i),'CreateGroup']
    df = df.merge(temp,on=['user_id','CreateGroup'],how = 'left')
    f.append('buycnt_{}'.format(i))


df['daycnt'] = df['o_day_series_o_sum']/df['o_day_series_o_ave']
temp = df[['user_id','daycnt','CreateGroup']][:]
temp.CreateGroup -=  30
for i in range(1,9):
    temp['CreateGroup'] = temp['CreateGroup'] + 30
    temp.columns = ['user_id','daycnt_{}'.format(i),'CreateGroup']
    df = df.merge(temp,on=['user_id','CreateGroup'],how = 'left')
    f.append('daycnt_{}'.format(i))


temp = df[['user_id','price_o_sum','CreateGroup']][:]
temp.CreateGroup -=  30
for i in range(1,9):
    temp['CreateGroup'] = temp['CreateGroup'] + 30
    temp.columns = ['user_id','price_o_sum_{}'.format(i),'CreateGroup']
    df = df.merge(temp,on=['user_id','CreateGroup'],how = 'left')
    f.append('price_o_sum_{}'.format(i))

temp = df[['user_id','daycnt','CreateGroup']][:]
for i in range(1,2):
    temp['CreateGroup'] = temp['CreateGroup'] - 30
    temp.columns = ['user_id','dc','CreateGroup']
    df = df.merge(temp,on=['user_id','CreateGroup'],how = 'left')
    f.append('dc')

df['last3'] = df['buy_1'] + df['buy_2'] + df['buy_3']
df['last5'] = df['buy_1'] + df['buy_2'] + df['buy_3'] + df['buy_4'] + df['buy_5']
df['last8'] = df['buy_1'] + df['buy_2'] + df['buy_3'] + df['buy_4'] + df['buy_5'] + df['buy_6'] + df['buy_7'] + df['buy_8']
for i in range(1,9):
    df.ix[df['buy_{}'.format(i)] == 0,'nextbuy_{}'.format(i)] = np.nan
day = 366
all = []
for i in range(9):
    temp_user = target_order[(target_order.o_day_series < day-30*i) & (target_order.o_day_series >= day - 90-30*i)][['user_id']].drop_duplicates().reset_index(drop=True)
    temp_user['CreateGroup'] = day-30*i
    all.append(temp_user)
temp_user = pd.concat(all).reset_index(drop=True)
df = temp_user.merge(df,on=['user_id','CreateGroup'],how = 'left')
#df[['user_id','CreateGroup','buy_1','buy_2','buy_3','last3'].to_csv('last3.csv',index = None)
df[f].to_csv('last3b.csv',index = None)

'''
T = []
dur = 187
for i in range(11):
    CG = day - 31*i
    temp = df[df.CreateGroup < CG][df.CreateGroup >= CG-dur].groupby(by = ['buy_1','buy_2','buy_3'])['buy'].mean().reset_index()
    temp.columns = ['buy_1','buy_2','buy_3','buy_mean']
    temp['CreateGroup'] = CG
    T.append(temp)
temp = pd.concat(T).reset_index(drop=True)
df = df.merge(temp,on=['buy_1','buy_2','buy_3','CreateGroup'],how = 'left')
T = []
for i in range(11):
    CG = day - 31*i
    temp = df[df.CreateGroup < CG][df.CreateGroup >= CG-dur].groupby(by = ['buy_1','buy_2','buy_3'])['buy'].sum().reset_index()
    temp.columns = ['buy_1','buy_2','buy_3','buy_sum']
    temp['CreateGroup'] = CG
    T.append(temp)
temp = pd.concat(T).reset_index(drop=True)
df = df.merge(temp,on=['buy_1','buy_2','buy_3','CreateGroup'],how = 'left')
T = []
for i in range(11):
    CG = day - 31*i
    temp = df[df.CreateGroup < CG][df.CreateGroup >= CG-dur].groupby(by = ['buy_1','buy_2','buy_3'])['buy'].std().reset_index()
    temp.columns = ['buy_1','buy_2','buy_3','buy_std']
    temp['CreateGroup'] = CG
    T.append(temp)
temp = pd.concat(T).reset_index(drop=True)
df = df.merge(temp,on=['buy_1','buy_2','buy_3','CreateGroup'],how = 'left')
T = []
for i in range(11):
    CG = day - 31*i
    temp = df[df.CreateGroup < CG][df.CreateGroup >= CG-dur].groupby(by = ['buy_1','buy_2','buy_3'])['buy'].count().reset_index()
    temp.columns = ['buy_1','buy_2','buy_3','buy_cnt']
    temp['CreateGroup'] = CG
    T.append(temp)
temp = pd.concat(T).reset_index(drop=True)
df = df.merge(temp,on=['buy_1','buy_2','buy_3','CreateGroup'],how = 'left')

fea = ['buy_mean','buy_sum','buy_std','buy_cnt']
for f in ['age_x','sex_x','user_lv_cd_x','buycnt','o_monthcnt','a_monthcnt','a_type','a_num','para_2_x','para_3_x','para_2_y','para_3_y','score_level']:
    print(f)
#    print(df.shape)
    T = []
    for i in range(11):
        CG = day - 31*i
        temp = df[df.CreateGroup < CG][df.CreateGroup >= CG-dur].groupby([f])['buy'].mean().reset_index()
        temp.columns = [f,'{}_mean'.format(f)]
        temp['CreateGroup'] = CG
#        print(temp)
        T.append(temp)
    fea.append('{}_mean'.format(f))
    temp = pd.concat(T).reset_index(drop=True)
    df = df.merge(temp,on=[f,'CreateGroup'],how = 'left')
#    print(df.shape)
    T = []
    for i in range(11):
        CG = day - 31*i
        temp = df[df.CreateGroup < CG][df.CreateGroup >= CG-dur].groupby([f])['buy'].std().reset_index()
        temp.columns = [f,'{}_std'.format(f)]
        temp['CreateGroup'] = CG
        T.append(temp)
    fea.append('{}_std'.format(f))
    temp = pd.concat(T).reset_index(drop=True)
    df = df.merge(temp,on=[f,'CreateGroup'],how = 'left')
#    print(df.shape)
    T = []
    for i in range(11):
        CG = day - 31*i
        temp = df[df.CreateGroup < CG][df.CreateGroup >= CG-dur].groupby([f])['buy'].skew().reset_index()
        temp.columns = [f,'{}_skew'.format(f)]
        temp['CreateGroup'] = CG
        T.append(temp)
#        print(temp)
    fea.append('{}_skew'.format(f))
    temp = pd.concat(T).reset_index(drop=True)
    df = df.merge(temp,on=[f,'CreateGroup'],how = 'left')
#    print(df.shape)
    T = []
    for i in range(11):
        CG = day - 31*i
        temp = df[df.CreateGroup < CG][df.CreateGroup >= CG-dur].groupby([f])['buy'].sum().reset_index()
        temp.columns = [f,'{}_sum'.format(f)]
        temp['CreateGroup'] = CG
        T.append(temp)
    fea.append('{}_sum'.format(f))
    temp = pd.concat(T).reset_index(drop=True)
    df = df.merge(temp,on=[f,'CreateGroup'],how = 'left')
#    print(df.shape)
    T = []
    for i in range(11):
        CG = day - 31*i
        temp = df[df.CreateGroup < CG][df.CreateGroup >= CG-dur].groupby([f])['buy'].count().reset_index()
        temp.columns = [f,'{}_cnt'.format(f)]
        temp['CreateGroup'] = CG
        T.append(temp)
    fea.append('{}_cnt'.format(f))
    temp = pd.concat(T).reset_index(drop=True)
    df = df.merge(temp,on=[f,'CreateGroup'],how = 'left')

df[['user_id','CreateGroup','buy_1','buy_2','buy_3','last3'] + fea].to_csv('last3_1.csv',index = None)
'''

