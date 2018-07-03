import pandas as pd
import numpy as np
import datetime

sku = pd.read_csv('B/jdata_sku_basic_info.csv', )
action = pd.read_csv('B/jdata_user_action.csv', parse_dates=['a_date'])
basic_info = pd.read_csv('B/jdata_user_basic_info.csv')
comment_score = pd.read_csv('B/jdata_user_comment_score.csv', parse_dates=['comment_create_tm'])
comment_score = comment_score.sort_values(by = 'comment_create_tm').reset_index(drop = True)
order = pd.read_csv('B/jdata_user_order.csv', parse_dates=['o_date'])

order = pd.merge(order, comment_score.drop_duplicates(subset =  ['user_id','o_id'],keep='last'), on=['user_id','o_id'], how = 'left')

order = pd.merge(order, sku, on='sku_id', how='left')
action = pd.merge(action, sku, how='left', on='sku_id')

## 换成时间序列
order['o_month_series'] = pd.to_datetime(order['o_date']).dt.month + (pd.to_datetime(order['o_date']).dt.year - 2016) * 12 - 9
action['a_month_series'] = pd.to_datetime(action['a_date']).dt.month + (pd.to_datetime(action['a_date']).dt.year - 2016) * 12 - 9

first_day = datetime.datetime.strptime('2016-08-31 00:00:00', '%Y-%m-%d %H:%M:%S')
order['c_day_series'] = (order['comment_create_tm'] - first_day).apply(lambda x: x.days)
order['o_day_series'] = (order['o_date'] - first_day).apply(lambda x: x.days)
action['a_day_series'] = (action['a_date'] - first_day).apply(lambda x: x.days)

##对用户行为提取特征
def ActionFeatures(Startday, PrepareDays, PredictDays, temp, dftemp):
    tempfeature = temp[temp.a_day_series < Startday][temp.a_day_series >= Startday-PrepareDays].reset_index(drop=True)
    templabel = temp[temp.a_day_series >= Startday][temp.a_day_series < Startday+PredictDays].reset_index(drop=True)
    dftemp = pd.merge(dftemp, templabel[['user_id','a_date']].drop_duplicates(subset = 'user_id', keep='last'), on = 'user_id',how='left').fillna(0)
    Checkcnt = tempfeature[['user_id','a_date']].groupby(['user_id']).count().reset_index()
    Checkcnt.columns = ['user_id', 'checkcnt']
    dftemp = pd.merge(dftemp,Checkcnt, how = 'left', on = 'user_id')
    monthcnt = tempfeature[['user_id','a_month_series']].drop_duplicates().groupby(['user_id']).size().reset_index()
    monthcnt.columns = ['user_id', 'a_monthcnt']
    dftemp = pd.merge(dftemp,monthcnt, how = 'left', on = 'user_id')
    tempfeature['daybeforelastcheck'] = tempfeature.sort_values(by=['user_id','a_day_series']).a_day_series - tempfeature.sort_values(by=['user_id','a_day_series']).groupby(['user_id']).shift(1).a_day_series
    for f in ['daybeforelastcheck', 'price', 'para_1', 'para_2', 'para_3', 'a_num','a_type','a_month_series','a_day_series']:
        a = tempfeature[['user_id',f]].groupby(['user_id']).mean().reset_index()
        a.columns = ['user_id', '{}_a_ave'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        a = tempfeature[['user_id',f]].groupby(['user_id']).std().reset_index()
        a.columns = ['user_id', '{}_a_std'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        a = tempfeature[['user_id',f]].groupby(['user_id']).sum().reset_index()
        a.columns = ['user_id', '{}_a_sum'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        a = tempfeature[['user_id',f]].groupby(['user_id']).median().reset_index()
        a.columns = ['user_id', '{}_a_median'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        a = tempfeature[['user_id',f]].groupby(['user_id']).min().reset_index()
        a.columns = ['user_id', '{}_a_min'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        a = tempfeature[['user_id',f]].groupby(['user_id']).max().reset_index()
        a.columns = ['user_id', '{}_a_max'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
#        a = tempfeature[['user_id',f]].groupby(['user_id']).var().reset_index()
#        a.columns = ['user_id', '{}_a_var'.format(f)]
#        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        dftemp = pd.merge(dftemp, tempfeature[['user_id',f]].drop_duplicates(subset = 'user_id', keep = 'last'), how = 'left', on = 'user_id')
    dftemp['CreateGroup'] = Startday
#    dftemp['ViewWeekAhead_{}'.format(post)] = dftemp['a_day_series'] > (Startday - 7)
#    dftemp['ViewMonthAhead_{}'.format(post)] = dftemp['a_day_series'] > (Startday - 31)
    return dftemp

## 对订单行为提取特征
def OrderFeatures(Startday, PrepareDays, PredictDays, temp, dftemp):
    tempfeature = temp[temp.o_day_series < Startday][temp.o_day_series >= Startday-PrepareDays].reset_index(drop=True) #特征数据集
    templabel = temp[temp.o_day_series >= Startday][temp.o_day_series < Startday+PredictDays].reset_index(drop=True) #标签数据集
    templabel['buy'] = 1
    templabel['nextbuy'] = templabel['o_day_series'] - Startday
    dftemp = pd.merge(dftemp, templabel[['user_id','buy']].drop_duplicates(subset = 'user_id', keep='first'), on = 'user_id',how='left').fillna(0)
    dftemp = pd.merge(dftemp, templabel[['user_id','nextbuy']].drop_duplicates(subset = 'user_id', keep='first'), on = 'user_id',how='left').fillna(0)
    dftemp = pd.merge(dftemp, templabel[['user_id','o_date']].drop_duplicates(subset = 'user_id', keep='first'), on = 'user_id',how='left').fillna(0)
    Buycnt = tempfeature[['user_id','o_id']].groupby(['user_id']).count().reset_index()
    Buycnt.columns = ['user_id', 'buycnt']
    dftemp = pd.merge(dftemp,Buycnt, how = 'left', on = 'user_id')
    monthcnt = tempfeature[['user_id','o_month_series']].drop_duplicates().groupby(['user_id']).size().reset_index()
    monthcnt.columns = ['user_id', 'o_monthcnt']
    dftemp = pd.merge(dftemp,monthcnt, how = 'left', on = 'user_id')
    print(tempfeature.shape)
#    print(tempfeature.sort_values(by=['user_id','o_day_series']).o_day_series)
    tempfeature['daybeforelastbuy'] = tempfeature.sort_values(by=['user_id','o_day_series']).o_day_series - tempfeature.sort_values(by=['user_id','o_day_series']).groupby(['user_id']).shift(1).o_day_series
    tempfeature['dayafterlastbuy'] = tempfeature.sort_values(by=['user_id','o_day_series']).groupby(['user_id']).shift(-1).o_day_series - tempfeature.sort_values(by=['user_id','o_day_series']).o_day_series
    tempfeature['itemperday'] = tempfeature['o_sku_num'] / tempfeature['dayafterlastbuy']
    tempfeature['itemperday_before'] = tempfeature['o_sku_num'] / tempfeature['daybeforelastbuy']
    for f in ['daybeforelastbuy', 'price', 'para_1', 'para_2', 'para_3', 'o_sku_num','o_month_series','o_day_series','c_day_series','score_level','itemperday','itemperday_before']: #对这批基础特征进行统计特征提取
        a = tempfeature[['user_id',f]].groupby(['user_id']).mean().reset_index()
        a.columns = ['user_id', '{}_o_ave'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        a = tempfeature[['user_id',f]].groupby(['user_id']).std().reset_index()
        a.columns = ['user_id', '{}_o_std'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        a = tempfeature[['user_id',f]].groupby(['user_id']).sum().reset_index()
        a.columns = ['user_id', '{}_o_sum'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        a = tempfeature[['user_id',f]].groupby(['user_id']).median().reset_index()
        a.columns = ['user_id', '{}_o_median'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        a = tempfeature[['user_id',f]].groupby(['user_id']).min().reset_index()
        a.columns = ['user_id', '{}_o_min'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        a = tempfeature[['user_id',f]].groupby(['user_id']).max().reset_index()
        a.columns = ['user_id', '{}_o_max'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
#        a = tempfeature[['user_id',f]].groupby(['user_id']).var().reset_index()
#        a.columns = ['user_id', '{}_o_var'.format(f)]
#        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        dftemp = pd.merge(dftemp, tempfeature[['user_id',f]].drop_duplicates(subset = 'user_id', keep = 'last'), how = 'left', on = 'user_id')
    dftemp['CreateGroup'] = Startday #表明是哪一个月份的
#    dftemp['OrderWeekAhead'] = dftemp['o_day_series'] > (Startday - 7)
#    dftemp['OrderMonthAhead'] = dftemp['o_day_series'] > (Startday - 31)
    return dftemp

## 对订单和用户行为提取特征
def OrderActionFeature(df, PredictDays):
    df['OrderActionDayDifference'] = df.a_day_series - df.o_day_series
    df['OrderCommentDayDifference'] = df.c_day_series - df.o_day_series
    df['ActionCommentDayDifference'] = df.c_day_series - df.a_day_series
    df['OrderEndDateDifference'] = df.CreateGroup - df.o_day_series
    df['ActionEndDateDifference'] = df.CreateGroup - df.a_day_series
    df['CommentEndDateDifference'] = df.CreateGroup - df.c_day_series
    df['OrderPluxAveInRange'] = df['OrderEndDateDifference'] - df['daybeforelastbuy_o_ave']
    df['ActionPluxAveInRange'] = df['ActionEndDateDifference'] - df['daybeforelastcheck_a_ave']
    df['PredictDays'] = PredictDays
    return df

## 只针对30和101商品，其他不管
o_temp = order[(order.cate == 30) | (order.cate == 101)].reset_index(drop=True)
o_temp = o_temp.sort_values(by=['user_id','o_day_series']).reset_index(drop=True)
#o_temp['nextbuy'] = o_temp.sort_values(by=['user_id','o_day_series']).groupby(['user_id']).shift(-1).o_day_series - o_temp.sort_values(by=['user_id','o_day_series']).o_day_series
a_temp = action[(action.cate == 30) | (action.cate == 101)].reset_index(drop=True)
a_temp = a_temp.sort_values(by=['user_id','a_day_series']).reset_index(drop=True)

premonth = 1
PrepareDays = 30*premonth #can use next 30 days as index to in
PredictDays = 30 #预测未来31天
Endday = 367 #最后一天
step = 30 #每个31天取一批次样本
monthnum = 9-premonth
o_df = []
a_df = []
#SDL = [242,258,273,289,304,320,335]

for Startday in range(156, Endday - PredictDays, step):
    print('creating dat4set from {} day'.format(Startday))
    o_df.append(OrderFeatures(Startday, PrepareDays, PredictDays, o_temp, basic_info[:]))
    a_df.append(ActionFeatures(Startday, PrepareDays, PredictDays, a_temp, basic_info[:]))
o_df = pd.concat(o_df).reset_index(drop=True)
a_df = pd.concat(a_df).reset_index(drop=True)
traindf = OrderActionFeature(pd.merge(o_df, a_df,on = ['user_id', 'CreateGroup'], how = 'left'), PredictDays)

traindf.to_csv('trainb_{}month_final.csv'.format(premonth), index = None)

o_df = OrderFeatures(Endday-1, PrepareDays, PredictDays, o_temp, basic_info[:])
a_df = ActionFeatures(Endday-1, PrepareDays, PredictDays, a_temp, basic_info[:])
predf = OrderActionFeature(pd.merge(o_df, a_df,on = ['user_id', 'CreateGroup'], how = 'left'), PredictDays)

predf.to_csv('testb_{}month_final.csv'.format(premonth), index = None)

## 只针对非目标商品，其他不管
o_temp = order[(order.cate == 1) | (order.cate == 46) | (order.cate == 71) | (order.cate == 83)].reset_index(drop=True)
o_temp = o_temp.sort_values(by=['user_id','o_day_series']).reset_index(drop=True)
#o_temp['nextbuy'] = o_temp.sort_values(by=['user_id','o_day_series']).groupby(['user_id']).shift(-1).o_day_series - o_temp.sort_values(by=['user_id','o_day_series']).o_day_series
a_temp = action[(action.cate == 1) | (action.cate == 46) | (action.cate == 71) | (action.cate == 83)].reset_index(drop=True)
a_temp = a_temp.sort_values(by=['user_id','a_day_series']).reset_index(drop=True)

#PrepareDays = 273 #can use next 30 days as index to increase the training size
#PredictDays = 31 #预测未来31天
#Endday = 367 #最后一天
#step = 31 #每个31天取一批次样本
o_df = []
a_df = []
for Startday in range(156, Endday - PredictDays, step):
    print('creating dataset from {} day'.format(Startday))
    o_df.append(OrderFeatures(Startday, PrepareDays, PredictDays, o_temp, basic_info[:]))
    a_df.append(ActionFeatures(Startday, PrepareDays, PredictDays, a_temp, basic_info[:]))
o_df = pd.concat(o_df).reset_index(drop=True)
a_df = pd.concat(a_df).reset_index(drop=True)
traindf = OrderActionFeature(pd.merge(o_df, a_df,on = ['user_id', 'CreateGroup'], how = 'left'), PredictDays)

traindf.to_csv('trainb_{}month_other_final.csv'.format(premonth), index = None)

o_df = OrderFeatures(Endday-1, PrepareDays, PredictDays, o_temp, basic_info[:])
a_df = ActionFeatures(Endday-1, PrepareDays, PredictDays, a_temp, basic_info[:])
predf = OrderActionFeature(pd.merge(o_df, a_df,on = ['user_id', 'CreateGroup'], how = 'left'), PredictDays)

predf.to_csv('testb_{}month_other_final.csv'.format(premonth), index = None)

