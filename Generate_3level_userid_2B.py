import pandas as pd
import numpy as np
import datetime

sku = pd.read_csv('./B/jdata_sku_basic_info.csv', )
action = pd.read_csv('./B/jdata_user_action.csv', parse_dates=['a_date'])
basic_info = pd.read_csv('./B/jdata_user_basic_info.csv')
comment_score = pd.read_csv('./B/jdata_user_comment_score.csv', parse_dates=['comment_create_tm'])
comment_score = comment_score.sort_values(by = 'comment_create_tm').reset_index(drop = True)
order = pd.read_csv('./B/jdata_user_order.csv', parse_dates=['o_date'])
sku.ix[sku.para_2 == -1,'para_2'] = np.nan
sku.ix[sku.para_3 == -1,'para_3'] = np.nan
order = pd.merge(order, comment_score.drop_duplicates(subset =  ['user_id','o_id'],keep='last'), on=['user_id','o_id'], how = 'left')

order = pd.merge(order, sku, on='sku_id', how='left')
action = pd.merge(action, sku, how='left', on='sku_id')

## 换成时间序列
order['o_month_series'] = pd.to_datetime(order['o_date']).dt.month + (pd.to_datetime(order['o_date']).dt.year - 2016) * 12 - 9
action['a_month_series'] = pd.to_datetime(action['a_date']).dt.month + (pd.to_datetime(action['a_date']).dt.year - 2016) * 12 - 9
#last3 = pd.read_csv('last3.csv')
#last3.CreateGroup = last3.CreateGroup + 31
first_day = datetime.datetime.strptime('2016-08-31 00:00:00', '%Y-%m-%d %H:%M:%S')
order['c_day_series'] = (order['comment_create_tm'] - first_day).apply(lambda x: x.days)
order['o_day_series'] = (order['o_date'] - first_day).apply(lambda x: x.days)
action['a_day_series'] = (action['a_date'] - first_day).apply(lambda x: x.days)
target_order = order[(order.cate == 101) | (order.cate == 30)].reset_index(drop=True)

##对用户行为提取特征
def ActionFeatures(Startday, PrepareDays, PredictDays, temp, dftemp):
    tempfeature = temp[temp.a_day_series < Startday][temp.a_day_series >= Startday-PrepareDays].reset_index(drop=True)
    tempfeature = tempfeature.merge(basic_info,on='user_id', how = 'left')
	#get target group
    temp_user = target_order[(target_order.o_day_series < Startday) & (target_order.o_day_series >= Startday - 90)][['user_id']].drop_duplicates().reset_index(drop=True)
    temp_user['act'] = 1
    tempfeature = tempfeature.merge(temp_user[['user_id','act']], on = 'user_id', how = 'left')
    tempfeature = tempfeature[~pd.isnull(tempfeature['act'])].reset_index(drop=True)
#    dftemp = dftemp.merge(temp_user[['user_id','act']], on = 'user_id', how = 'left')
#    dftemp = dftemp[~pd.isnull(dftemp['act'])].reset_index(drop=True)

    dftemp = pd.merge(dftemp, tempfeature[['user_id','cate','sku_id']].drop_duplicates(subset = 'user_id', keep='last'), on = 'user_id',how='left').fillna(0)
    templabel = temp[temp.a_day_series >= Startday][temp.a_day_series < Startday+PredictDays].reset_index(drop=True)
    dftemp = pd.merge(dftemp, templabel[['user_id','a_date']].drop_duplicates(subset = 'user_id', keep='first'), on = 'user_id',how='left').fillna(0)
    Checkcnt = tempfeature[['user_id','a_date']].groupby(['user_id']).count().reset_index()
    Checkcnt.columns = ['user_id', 'checkcnt']
    dftemp = pd.merge(dftemp,Checkcnt, how = 'left', on = 'user_id')
    monthcnt = tempfeature[['user_id','a_month_series']].drop_duplicates().groupby(['user_id']).size().reset_index()
    monthcnt.columns = ['user_id', 'a_monthcnt']
    dftemp = pd.merge(dftemp,monthcnt, how = 'left', on = 'user_id')
    tempfeature['daybeforelastcheck'] = tempfeature.sort_values(by=['user_id','a_day_series']).a_day_series - tempfeature.sort_values(by=['user_id','a_day_series']).groupby(['user_id']).shift(1).a_day_series

    for j in ['sku_id','user_lv_cd','a_month_series','a_day_series']:
        Buycnt = tempfeature[[j,'a_date']].groupby([j]).count().reset_index()
        Buycnt.columns = [j, 'checkcnt_a_{}'.format(j)]
        tempfeature = pd.merge(tempfeature,Buycnt, how = 'left', on = j)
        for f in ['daybeforelastcheck']:
            a = tempfeature[[j,f]].groupby([j]).mean().reset_index()
            a.columns = [j, '{}_a_ave_{}'.format(f,j)]
            tempfeature = tempfeature.merge(a,how = 'left', on = j)

    tempfeature['a_month_series'] = tempfeature['a_month_series'].astype(int)
    tempfeature['a_day_series'] = tempfeature['a_day_series'].astype(int)

    for j in ['cate']:
        Buycnt = tempfeature[[j,'a_date']].groupby([j]).count().reset_index()
        Buycnt.columns = [j, 'checkcnt_a_{}'.format(j)]
        tempfeature = pd.merge(tempfeature,Buycnt, how = 'left', on = j)
        for f in ['daybeforelastcheck','price', 'para_1', 'para_2', 'para_3']:
            a = tempfeature[[j,f]].groupby([j]).mean().reset_index()
            a.columns = [j, '{}_a_ave_{}'.format(f,j)]
            tempfeature = tempfeature.merge(a,how = 'left', on = j)

    print(tempfeature.shape)
    a = tempfeature[['a_num','cate','daybeforelastcheck']].groupby(['a_num','cate']).mean().reset_index()
    a.columns = ['a_num','cate','daybeforelastcheck_a_num_cate']
    tempfeature = tempfeature.merge(a,how = 'left', on = ['a_num','cate'])

    a = tempfeature[['a_num','cate','a_month_series','daybeforelastcheck']].groupby(['a_num','cate']).mean().reset_index()
    a.columns = ['a_num','cate','a_month_series','daybeforelastcheck_a_num_cate_a_month_series']
    tempfeature = tempfeature.merge(a,how = 'left', on = ['a_num','cate','a_month_series'])
    tempfeature['a_month_series'] = tempfeature['a_month_series'].astype(int)
    tempfeature['a_day_series'] = tempfeature['a_day_series'].astype(int)
    print(tempfeature.shape)

    for f in ['checkcnt_a_sku_id','checkcnt_a_cate','daybeforelastcheck_a_ave_sku_id','daybeforelastcheck_a_ave_cate','daybeforelastcheck',
              'price_a_ave_cate', 'para_1_a_ave_cate', 'daybeforelastcheck_a_ave_user_lv_cd','daybeforelastcheck_a_ave_a_month_series',
              'daybeforelastcheck_a_ave_a_day_series','daybeforelastcheck_a_num_cate_a_month_series','daybeforelastcheck_a_num_cate',
              'price', 'para_1', 'para_2', 'para_3', 'a_num','a_type','a_month_series','a_day_series']:
        a = tempfeature[['user_id',f]].groupby(['user_id']).mean().reset_index()
        a.columns = ['user_id', '{}_a_ave'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        tempfeature = tempfeature.merge(a,how = 'left', on = 'user_id')
        a = tempfeature[['user_id',f]].groupby(['user_id']).std().reset_index()
        a.columns = ['user_id', '{}_a_std'.format(f)]
        dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        if 'ave' not in f:
            a = tempfeature[['user_id',f]].groupby(['user_id']).sum().reset_index()
            a.columns = ['user_id', '{}_a_sum'.format(f)]
            dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
        if ('user_lv_cd' not in f) | ('a_ave_a_month_series' not in f) | ('a_ave_a_day_series' not in f):
            a = tempfeature[['user_id',f]].groupby(['user_id']).min().reset_index()
            a.columns = ['user_id', '{}_a_min'.format(f)]
            dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
            a = tempfeature[['user_id',f]].groupby(['user_id']).max().reset_index()
            a.columns = ['user_id', '{}_a_max'.format(f)]
            dftemp = pd.merge(dftemp,a, how = 'left', on = 'user_id')
            dftemp = pd.merge(dftemp, tempfeature[['user_id',f]].drop_duplicates(subset = 'user_id', keep = 'last'), how = 'left', on = 'user_id')

    for j in ['sku_id','cate','a_month_series']:
        for f in ['daybeforelastcheck_a_ave']:
            a = tempfeature[[j,f]].groupby([j]).mean().reset_index()
            a.columns = [j, '{}_o_ave_{}'.format(f,j)]
            dftemp = dftemp.merge(a,how = 'left', on = j)

    dftemp['CreateGroup'] = Startday
#    dftemp['ViewWeekAhead_{}'.format(post)] = dftemp['a_day_series'] > (Startday - 7)
#    dftemp['ViewMonthAhead_{}'.format(post)] = dftemp['a_day_series'] > (Startday - 31)
    return dftemp

## 对订单行为提取特征
def OrderFeatures(Startday, PrepareDays, PredictDays, temp, dftemp):
    tempfeature = temp[temp.o_day_series < Startday][temp.o_day_series >= Startday-PrepareDays].reset_index(drop=True) #特征数据集
    tempfeature = tempfeature.merge(basic_info,on='user_id', how = 'left')
    print(list(tempfeature.columns))
    templabel = temp[temp.o_day_series >= Startday][temp.o_day_series < Startday+PredictDays].reset_index(drop=True) #标签数据集
    #get target group
    print(tempfeature.shape)
    temp_user = target_order[(target_order.o_day_series < Startday) & (target_order.o_day_series >= Startday - 90)][['user_id']].drop_duplicates().reset_index(drop=True)
    temp_user['act'] = 1
    tempfeature = tempfeature.merge(temp_user[['user_id','act']], on = 'user_id', how = 'left')
    tempfeature = tempfeature[~pd.isnull(tempfeature['act'])].reset_index(drop=True)
#    dftemp = dftemp.merge(temp_user[['user_id','act']], on = 'user_id', how = 'left')
#    dftemp = dftemp[~pd.isnull(dftemp['act'])].reset_index(drop=True)

    templabel['buy'] = 1
    templabel['nextbuy'] = templabel['o_day_series'] - Startday
    dftemp = pd.merge(dftemp, templabel[['user_id','buy']].drop_duplicates(subset = 'user_id', keep='first'), on = 'user_id',how='left').fillna(0)
    dftemp = pd.merge(dftemp, templabel[['user_id','nextbuy']].drop_duplicates(subset = 'user_id', keep='first'), on = 'user_id',how='left').fillna(0)
    dftemp = pd.merge(dftemp, templabel[['user_id','o_date']].drop_duplicates(subset = 'user_id', keep='first'), on = 'user_id',how='left').fillna(0)
    dftemp = pd.merge(dftemp, tempfeature[['user_id','cate','sku_id']].drop_duplicates(subset = 'user_id', keep='last'), on = 'user_id',how='left').fillna(0)
    dftemp = pd.merge(dftemp, tempfeature[['user_id','o_area']].drop_duplicates(subset = 'user_id', keep='last'), on = 'user_id',how='left').fillna(0)
    monthcnt = tempfeature[['user_id','o_month_series']].drop_duplicates().groupby(['user_id']).size().reset_index()
    monthcnt.columns = ['user_id', 'o_monthcnt']
    dftemp = pd.merge(dftemp,monthcnt, how = 'left', on = 'user_id')
    print(tempfeature.shape)
    tempfeature['daybeforelastbuy'] = tempfeature.sort_values(by=['user_id','o_day_series']).o_day_series - tempfeature.sort_values(by=['user_id','o_day_series']).groupby(['user_id']).shift(1).o_day_series
    tempfeature['dayafterlastbuy'] = tempfeature.sort_values(by=['user_id','o_day_series']).groupby(['user_id']).shift(-1).o_day_series - tempfeature.sort_values(by=['user_id','o_day_series']).o_day_series
    tempfeature['itemperday'] = tempfeature['o_sku_num'] / tempfeature['dayafterlastbuy']
    tempfeature['itemperday_before'] = tempfeature['o_sku_num'] / tempfeature['daybeforelastbuy']
    print(list(dftemp.columns))

    dftemp['CreateGroup'] = Startday
#    dftemp = dftemp.merge(last3[['user_id','CreateGroup','nextbuy_1']],on=['user_id','CreateGroup'],how = 'left')

    for j in ['sku_id']:
        Buycnt = tempfeature[[j,'o_id']].groupby([j]).count().reset_index()
        Buycnt.columns = [j, 'buycnt_o_{}'.format(j)]
        tempfeature = pd.merge(tempfeature,Buycnt, how = 'left', on = j)
        for f in ['daybeforelastbuy', 'o_sku_num','score_level','itemperday_before']:
            a = tempfeature[[j,f]].groupby([j]).mean().reset_index()
            a.columns = [j, '{}_o_ave_o_{}'.format(f,j)]
            b = tempfeature[[j,f]].groupby([j]).count().reset_index()
            tempfeature = tempfeature.merge(a[b[j] > 10],how = 'left', on = j)

    for j in ['user_lv_cd','o_month_series','o_day_series']:
        Buycnt = tempfeature[[j,'o_id']].groupby([j]).count().reset_index()
        Buycnt.columns = [j, 'buycnt_o_{}'.format(j)]
        tempfeature = pd.merge(tempfeature,Buycnt, how = 'left', on = j)
        for f in ['daybeforelastbuy']:
            a = tempfeature[[j,f]].groupby([j]).mean().reset_index()
            a.columns = [j, '{}_o_ave_o_{}'.format(f,j)]
            tempfeature = tempfeature.merge(a,how = 'left', on = j)
    print(tempfeature.shape)
    a = tempfeature[['o_sku_num','cate','daybeforelastbuy']].groupby(['o_sku_num','cate']).mean().reset_index()
    a.columns = ['o_sku_num','cate','daybeforelastbuy_o_sku_num_cate']
    tempfeature = tempfeature.merge(a,how = 'left', on = ['o_sku_num','cate'])

    a = tempfeature[['o_sku_num','cate','o_month_series','daybeforelastbuy']].groupby(['o_sku_num','cate']).mean().reset_index()
    a.columns = ['o_sku_num','cate','o_month_series','daybeforelastbuy_o_sku_num_cate_o_month_series']
    tempfeature = tempfeature.merge(a,how = 'left', on = ['o_sku_num','cate','o_month_series'])
    tempfeature['o_month_series'] = tempfeature['o_month_series'].astype(int)
    tempfeature['o_day_series'] = tempfeature['o_day_series'].astype(int)
    print(tempfeature.shape)

    for j in ['cate']:
        Buycnt = tempfeature[[j,'o_id']].groupby([j]).count().reset_index()
        Buycnt.columns = [j, 'buycnt_o_{}'.format(j)]
        tempfeature = pd.merge(tempfeature,Buycnt, how = 'left', on = j)
        for f in ['daybeforelastbuy', 'o_sku_num','itemperday','itemperday_before', 'price', 'para_1', 'para_2', 'para_3']:
            a = tempfeature[[j,f]].groupby([j]).mean().reset_index()
            a.columns = [j, '{}_o_ave_o_{}'.format(f,j)]
            tempfeature = tempfeature.merge(a,how = 'left', on = j)

    for j in ['user_id']:
        Buycnt = tempfeature[[j,'o_id']].groupby([j]).count().reset_index()
        Buycnt.columns = [j, 'buycnt_o_{}'.format(j)]
        dftemp = pd.merge(dftemp,Buycnt, how = 'left', on = j)
        for f in ['o_month_series','buycnt_o_sku_id','buycnt_o_cate','o_sku_num_o_ave_o_cate','daybeforelastbuy_o_ave_o_o_month_series',
                  'daybeforelastbuy_o_ave_o_user_lv_cd','daybeforelastbuy_o_sku_num_cate','daybeforelastbuy_o_ave_o_o_day_series',
                  'itemperday_o_ave_o_cate','daybeforelastbuy_o_ave_o_cate','o_sku_num_o_ave_o_sku_id',
                  'price_o_ave_o_cate', 'para_1_o_ave_o_cate','para_2_o_ave_o_cate','para_3_o_ave_o_cate',
                  'score_level_o_ave_o_sku_id','daybeforelastbuy_o_ave_o_sku_id','daybeforelastbuy_o_sku_num_cate_o_month_series',
                  'daybeforelastbuy', 'price', 'para_1', 'para_2', 'para_3', 'o_sku_num',
                  'o_day_series','c_day_series','score_level','itemperday','itemperday_before']:
            print(f)
            a = tempfeature[[j,f]].groupby([j]).mean().reset_index()
            a.columns = [j, '{}_o_ave_o_{}'.format(f,j)]
            dftemp = pd.merge(dftemp,a, how = 'left', on = j)
            tempfeature = tempfeature.merge(a,how = 'left', on = j)
            a = tempfeature[[j,f]].groupby([j]).std().reset_index()
            a.columns = [j, '{}_o_std_o_{}'.format(f,j)]
            dftemp = pd.merge(dftemp,a, how = 'left', on = j)
            if 'ave' not in f:
                a = tempfeature[[j,f]].groupby([j]).sum().reset_index()
                a.columns = [j, '{}_o_sum_o_{}'.format(f,j)]
                dftemp = pd.merge(dftemp,a, how = 'left', on = j)
            if ('user_lv_cd' not in f) | ('o_o_month_series' not in f) | ('o_sku_num_cate' not in f)| ('o_o_day_series' not in f):
                a = tempfeature[[j,f]].groupby([j]).min().reset_index()
                a.columns = [j, '{}_o_min_o_{}'.format(f,j)]
                dftemp = pd.merge(dftemp,a, how = 'left', on = j)
                a = tempfeature[[j,f]].groupby([j]).max().reset_index()
                a.columns = [j, '{}_o_max_o_{}'.format(f,j)]
                dftemp = pd.merge(dftemp,a, how = 'left', on = j)
                dftemp = pd.merge(dftemp, tempfeature[[j,f]].drop_duplicates(subset = j, keep = 'last'), how = 'left', on = j)

    for j in ['sku_id','cate','o_month_series']:
        for f in ['daybeforelastbuy_o_ave_o_user_id']:
            a = tempfeature[[j,f]].groupby([j]).mean().reset_index()
            a.columns = [j, '{}_o_ave_o_{}'.format(f,j)]
            dftemp = dftemp.merge(a,how = 'left', on = j)

#    dftemp['CreateGroup'] = Startday #表明是哪一个月份的
#    dftemp['CommentWeekAhead'] = dftemp['c_day_series'] > (Startday - 7)
#    dftemp['CommentMonthAhead'] = dftemp['c_day_series'] > (Startday - 31)
    return dftemp

## 对订单和用户行为提取特征
def OrderActionFeature(df, PredictDays):
    df['OrderActionDayDifference'] = df.a_day_series - df.o_day_series
    df['OrderCommentDayDifference'] = df.c_day_series - df.o_day_series
    df['ActionCommentDayDifference'] = df.c_day_series - df.a_day_series
    df['OrderEndDateDifference'] = df.CreateGroup - df.o_day_series
    df['ActionEndDateDifference'] = df.CreateGroup - df.a_day_series
    df['CommentEndDateDifference'] = df.CreateGroup - df.c_day_series
    df['OrderPluxAveInRange'] = df['OrderEndDateDifference'] - df['daybeforelastbuy_o_ave_o_user_id']
    df['ActionPluxAveInRange'] = df['ActionEndDateDifference'] - df['daybeforelastcheck_a_ave']
    df['OrderPluxAveInRange2'] = df['OrderEndDateDifference'] - df['daybeforelastbuy_o_ave_o_sku_id_o_ave_o_user_id']
    df['OrderPluxAveInRange25'] = df['OrderEndDateDifference'] - df['daybeforelastbuy_o_ave_o_cate_o_ave_o_user_id']
    df['ActionPluxAveInRange2'] = df['ActionEndDateDifference'] - df['daybeforelastcheck_a_ave_sku_id_a_ave']
    df['ActionPluxAveInRange25'] = df['ActionEndDateDifference'] - df['daybeforelastcheck_a_ave_cate_a_ave']
    df['OrderPluxAveInRange3'] = df['OrderEndDateDifference'] - df['daybeforelastbuy_o_ave_o_cate_o_ave_o_user_id']
    df['ActionPluxAveInRange3'] = df['ActionEndDateDifference'] - df['daybeforelastcheck_a_ave_cate_a_ave']
    df['OrderPluxAveInRange4'] = df['OrderEndDateDifference'] - df['daybeforelastbuy_o_ave_o_user_id_o_ave_o_sku_id']
    df['OrderPluxAveInRange5'] = df['OrderEndDateDifference'] - df['daybeforelastbuy_o_ave_o_user_id_o_ave_o_cate']
    df['OrderPluxAveInRange_user_lv_cd'] = df['OrderEndDateDifference'] - df['daybeforelastbuy_o_ave_o_user_lv_cd_o_ave_o_user_id']
    df['OrderPluxAveInRange_o_month_series'] = df['OrderEndDateDifference'] - df['daybeforelastbuy_o_ave_o_user_id_o_ave_o_o_month_series']
    df['OrderPluxAveInRange_o_month_series2'] = df['OrderEndDateDifference'] - df['daybeforelastbuy_o_ave_o_o_month_series_o_ave_o_user_id']
    df['OrderPluxAveInRange_daybeforelastbuy_o_sku_num_cate_o_month_series'] = df['OrderEndDateDifference'] - df['daybeforelastbuy_o_sku_num_cate_o_month_series_o_ave_o_user_id']
    df['OrderPluxAveInRange_daybeforelastbuy_o_sku_num_cate'] = df['OrderEndDateDifference'] - df['daybeforelastbuy_o_sku_num_cate_o_ave_o_user_id']
    df['PredictDays'] = PredictDays
    return df

## 只针对30和101商品，其他不管
o_temp = order[(order.cate == 30) | (order.cate == 101)].reset_index(drop=True)
o_temp = o_temp.sort_values(by=['user_id','o_day_series']).reset_index(drop=True)
a_temp = action[(action.cate == 30) | (action.cate == 101)].reset_index(drop=True)
a_temp = a_temp.sort_values(by=['user_id','a_day_series']).reset_index(drop=True)

premonth = 3
PrepareDays = 30*premonth #can use next 30 days as index to in$
PredictDays = 30 #预测未来31天
Endday = 367 #最后一天
step = 30 #每个31天取一批次样本
monthnum = 9-premonth
o_df = []
a_df = []
SDL = [242,258,273,289,304,320,335]

for Startday in range(216, Endday - PredictDays, step):
    print('creating dataset from {} day'.format(Startday))
    o_df.append(OrderFeatures(Startday, PrepareDays, PredictDays, o_temp, basic_info[:]))
    temp = o_temp[o_temp.o_day_series < Startday].drop_duplicates(subset = 'user_id', keep='last')
    temp = pd.merge(a_temp, temp[['user_id', 'o_day_series']], how = 'left', on = 'user_id')
    a_df.append(ActionFeatures(Startday, PrepareDays, PredictDays, temp, basic_info[:]))

o_df = pd.concat(o_df).reset_index(drop=True)
a_df = pd.concat(a_df).reset_index(drop=True)
traindf = OrderActionFeature(pd.merge(o_df, a_df,on = ['user_id', 'CreateGroup'], how = 'left'), PredictDays)

traindf.to_csv('trainb_{}month_3level_userid_2.csv'.format(premonth), index = None)

o_df = OrderFeatures(Endday-1, PrepareDays, PredictDays, o_temp, basic_info[:])
temp = o_temp[o_temp.o_day_series < Endday].drop_duplicates(subset = 'user_id', keep='last')
temp = pd.merge(a_temp, temp[['user_id', 'o_day_series']], how = 'left', on = 'user_id')
a_df = ActionFeatures(Endday-1, PrepareDays, PredictDays, temp, basic_info[:])
predf = OrderActionFeature(pd.merge(o_df, a_df,on = ['user_id', 'CreateGroup'], how = 'left'), PredictDays)

predf.to_csv('testb_{}month_3level_userid_2.csv'.format(premonth), index = None)

