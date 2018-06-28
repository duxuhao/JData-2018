import numpy as np
import pandas as pd
import datetime

def order_hist0(CreateGroupList):
    order = pd.read_csv('./B/jdata_user_order.csv', parse_dates=['o_date'])
    sku = pd.read_csv('./B/jdata_sku_basic_info.csv', )
    order = pd.merge(order, sku, on='sku_id', how='left')
    target_order = order[(order.cate == 101) | (order.cate == 30)].reset_index(drop=True)
    first_day = datetime.datetime.strptime('2016-08-31 00:00:00', '%Y-%m-%d %H:%M:%S')
    target_order['o_day_series'] = (target_order['o_date'] - first_day).apply(lambda x: x.days)
    basic_info = pd.read_csv('./B/jdata_user_basic_info.csv')

    target_order = target_order.sort_values(by=['user_id','o_day_series'], ascending=False).reset_index(drop=True)

    alld = []
    for CG in CreateGroupList:
        CreateGroup = CG
        t = target_order[target_order.o_day_series < CreateGroup]
        features =[]
        for i in range(10):
            t2 = t[['user_id','o_day_series']].groupby(['user_id']).shift(-i)
            t2.columns = t2.columns + '_{}'.format(i)
            features.append(t2.columns[0])
            t = pd.concat([t,t2],axis=1)
        x = t.drop_duplicates(subset=['user_id'])
        x = x[['user_id'] + features]
        x['CreateGroup'] = CreateGroup
        alld.append(x)
    return pd.concat(alld)

def all_order_hist(CreateGroupList,num,f):
    order = pd.read_csv('./B/jdata_user_order.csv', parse_dates=['o_date'])
    sku = pd.read_csv('./B/jdata_sku_basic_info.csv', )
    order = pd.merge(order, sku, on='sku_id', how='left')
    target_order = order.reset_index(drop=True)
    first_day = datetime.datetime.strptime('2016-08-31 00:00:00', '%Y-%m-%d %H:%M:%S')
    target_order['o_day_series'] = (target_order['o_date'] - first_day).apply(lambda x: x.days)

    target_order = target_order.sort_values(by=['user_id','o_day_series'], ascending=False).reset_index(drop=True)

    alld = []
    for CG in CreateGroupList:
        CreateGroup = CG
        t = target_order[target_order.o_day_series < CreateGroup]
        features =[]
        for i in range(num):
            t2 = t[['user_id',f]].groupby(['user_id']).shift(-i)
            t2.columns = t2.columns + '_all_{}'.format(i)
            features.append(t2.columns[0])
            t = pd.concat([t,t2],axis=1)
        x = t.drop_duplicates(subset=['user_id'])
        x = x[['user_id'] + features]
        x['CreateGroup'] = CreateGroup
        alld.append(x)
    df = pd.concat(alld).reset_index(drop=True)
#    print(np.unique(df.CreateGroup))
    return df

def order_price_total_hist(CreateGroupList,num):
    order = pd.read_csv('./B/jdata_user_order.csv', parse_dates=['o_date'])
    sku = pd.read_csv('./B/jdata_sku_basic_info.csv', )
    order = pd.merge(order, sku, on='sku_id', how='left')
    order['price_total'] = order['o_sku_num'] * order['price']
    target_order = order[(order.cate == 101) | (order.cate == 30)].reset_index(drop=True)
    first_day = datetime.datetime.strptime('2016-08-31 00:00:00', '%Y-%m-%d %H:%M:%S')
    target_order['o_day_series'] = (target_order['o_date'] - first_day).apply(lambda x: x.days)

    target_order = target_order.sort_values(by=['user_id','o_day_series'], ascending=False).reset_index(drop=True)

    alld = []
    for CG in CreateGroupList:
        CreateGroup = CG
        t = target_order[target_order.o_day_series < CreateGroup]
        features =[]
        for i in range(num):
            t2 = t[['user_id','price_total']].groupby(['user_id']).shift(-i)
            t2.columns = t2.columns + '_{}'.format(i)
            features.append(t2.columns[0])
            t = pd.concat([t,t2],axis=1)
        x = t.drop_duplicates(subset=['user_id'])
        x = x[['user_id'] + features]
        x['CreateGroup'] = CreateGroup
        alld.append(x)
    df = pd.concat(alld).reset_index(drop=True)
    return df

def order_hist(CreateGroupList,num,f):
    order = pd.read_csv('./B/jdata_user_order.csv', parse_dates=['o_date'])
    sku = pd.read_csv('./B/jdata_sku_basic_info.csv', )
    order = pd.merge(order, sku, on='sku_id', how='left')
    target_order = order[(order.cate == 101) | (order.cate == 30)].reset_index(drop=True)
    first_day = datetime.datetime.strptime('2016-08-31 00:00:00', '%Y-%m-%d %H:%M:%S')
    target_order['o_day_series'] = (target_order['o_date'] - first_day).apply(lambda x: x.days)

    target_order = target_order.sort_values(by=['user_id','o_day_series'], ascending=False).reset_index(drop=True)

    alld = []
    for CG in CreateGroupList:
        CreateGroup = CG
        t = target_order[target_order.o_day_series < CreateGroup]
        features =[]
        for i in range(num):
            t2 = t[['user_id',f]].groupby(['user_id']).shift(-i)
            t2.columns = t2.columns + '_{}'.format(i)
            features.append(t2.columns[0])
            t = pd.concat([t,t2],axis=1)
        x = t.drop_duplicates(subset=['user_id'])
        x = x[['user_id'] + features]
        x['CreateGroup'] = CreateGroup
        alld.append(x)
    df = pd.concat(alld).reset_index(drop=True)
#    print(np.unique(df.CreateGroup))
    return df

def action_hist(CreateGroupList,num,f):
    action = pd.read_csv('./B/jdata_user_action.csv', parse_dates=['a_date'])
    sku = pd.read_csv('./B/jdata_sku_basic_info.csv', )
    action = pd.merge(action, sku, on='sku_id', how='left')
    target_action = action[(action.cate == 101) | (action.cate == 30)].reset_index(drop=True)
    first_day = datetime.datetime.strptime('2016-08-31 00:00:00', '%Y-%m-%d %H:%M:%S')
    target_action['a_day_series'] = (target_action['a_date'] - first_day).apply(lambda x: x.days)

    target_action = target_action.sort_values(by=['user_id','a_day_series'], ascending=False).reset_index(drop=True)

    alld = []
    for CG in CreateGroupList:
        CreateGroup = CG
        t = target_action[target_action.a_day_series < CreateGroup]
        features =[]
        for i in range(num):
            t2 = t[['user_id',f]].groupby(['user_id']).shift(-i)
            t2.columns = t2.columns + '_{}'.format(i)
            features.append(t2.columns[0])
            t = pd.concat([t,t2],axis=1)
        x = t.drop_duplicates(subset=['user_id'])
        x = x[['user_id'] + features]
        x['CreateGroup'] = CreateGroup
        alld.append(x)
    df = pd.concat(alld).reset_index(drop=True)
    return df

def all_action_hist(CreateGroupList,num,f):
    action = pd.read_csv('./B/jdata_user_action.csv', parse_dates=['a_date'])
    sku = pd.read_csv('./B/jdata_sku_basic_info.csv', )
    action = pd.merge(action, sku, on='sku_id', how='left')
    target_action = action.reset_index(drop=True)
    first_day = datetime.datetime.strptime('2016-08-31 00:00:00', '%Y-%m-%d %H:%M:%S')
    target_action['a_day_series'] = (target_action['a_date'] - first_day).apply(lambda x: x.days)

    target_action = target_action.sort_values(by=['user_id','a_day_series'], ascending=False).reset_index(drop=True)

    alld = []
    for CG in CreateGroupList:
        CreateGroup = CG
        t = target_action[target_action.a_day_series < CreateGroup]
        features =[]
        for i in range(num):
            t2 = t[['user_id',f]].groupby(['user_id']).shift(-i)
            t2.columns = t2.columns + '_all_{}'.format(i)
            features.append(t2.columns[0])
            t = pd.concat([t,t2],axis=1)
        x = t.drop_duplicates(subset=['user_id'])
        x = x[['user_id'] + features]
        x['CreateGroup'] = CreateGroup
        alld.append(x)
    df = pd.concat(alld).reset_index(drop=True)
    return df
