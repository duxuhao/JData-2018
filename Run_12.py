from History_detailB import order_hist,order_price_total_hist,action_hist
from utility import Merge,readlog
import xgboost as xgb
import lightgbm as lgbm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

order = pd.read_csv('./B/jdata_user_order.csv', parse_dates=['o_date'])
sku = pd.read_csv('./B/jdata_sku_basic_info.csv', )
order = pd.merge(order, sku, on='sku_id', how='left')
target_order = order[(order.cate == 101) | (order.cate == 30)].reset_index(drop=True)
first_day = datetime.datetime.strptime('2016-08-31 00:00:00', '%Y-%m-%d %H:%M:%S')
target_order['o_day_series'] = (target_order['o_date'] - first_day).apply(lambda x: x.days)

def score(pred, real): #评分系统，感谢herhert，只要s1
    pred['index'] = np.arange(pred.shape[0]) + 1
    pred['wi'] = 1 / (1 + np.log(pred['index']))

    compare = pd.merge(pred, real, how='left', on='user_id')
    compare = compare[:40000]
    compare.fillna(0, inplace=True)
    S10 = np.sum(compare['buy'] * compare['wi']) / compare['wi'].sum()
    print(S10)
    compare = compare[:34000]
    S1 = np.sum(compare['buy'] * compare['wi']) / compare['wi'].sum()
    print(S1)
    return S1

def predict22(X_all, X_new, features, clf, score, v = False, esr=50, sk=3, fn='submission'):
    first_day = datetime.datetime.strptime('2017-08-31 00:00:00', '%Y-%m-%d %H:%M:%S')
    temp_user = target_order[(target_order.o_day_series < 336) & (target_order.o_day_series >= 274)][['user_id']].drop_duplicates().reset_index(drop=True)
    temp_user['CreateGroup'] = 336
    print('before delete: {}'.format(X_all.shape))
    X = temp_user.merge(X_all,on=['user_id','CreateGroup'],how = 'left')
    print('after delete: {}'.format(X.shape))
    temp_user = target_order[(target_order.o_day_series < 366) & \
                             (target_order.o_day_series >= 366 - 74)][['user_id']].drop_duplicates().reset_index(drop=True)
    temp_user['CreateGroup'] = 366
    print('before delete: {}'.format(X_new.shape))
    X_new = temp_user.merge(X_new,on=['user_id','CreateGroup'],how = 'left')
    temp_user = target_order[(target_order.o_day_series < 306) & (target_order.o_day_series >= 215)][['user_id']].drop_duplicates().reset_index(drop=True)
    temp_user['CreateGroup'] = 306
    print('before delete: {}'.format(X_all.shape))
    X2 = temp_user.merge(X_all,on=['user_id','CreateGroup'],how = 'left')
    print('Train: {}'.format(X_new.shape))
    kf = KFold(n_splits=sk)
    print(len(features))
    Performance = []
    X_new['Prob'] = 0
    X_new['Prob_x'] = 0
    X['Prob_x'] = 0
    for train_index, test_index in kf.split(X2):
        X_train, X_test = X2.ix[train_index,:], X2.ix[test_index,:]
        X_train, X_test = X_train[features], X_test[features]
        y_train, y_test = X2.ix[train_index,:].buy, X2.ix[test_index,:].buy
        clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='auc', verbose=v, early_stopping_rounds=esr)
        X_new['Prob_x'] = X_new['Prob_x'] + clf.predict_proba(X_new[features])[:,1]/sk
        X['Prob_x'] = X['Prob_x'] + clf.predict_proba(X[features])[:,1]/sk
    features.append('Prob_x')
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.ix[train_index,:], X.ix[test_index,:]
        X_train, X_test = X_train[features], X_test[features]
        y_train, y_test = X.ix[train_index,:].buy, X.ix[test_index,:].buy
        clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='auc', verbose=v, early_stopping_rounds=esr)
        pred = clf.predict_proba(X_test)[:,1]
        X_new['Prob'] = X_new['Prob'] + clf.predict_proba(X_new[features])[:,1]/sk
        Performance.append(roc_auc_score(y_test,pred))
    print("Mean Score: {}".format(np.mean(Performance)))
    X_new['Days'] = np.random.randint(15,size=len(X_new))
    X_new['pred_date'] = X_new['Days'].apply(lambda x: (datetime.timedelta(days=x) + first_day).strftime("%Y-%m-%d"))
    X_new.sort_values(by = ['Prob'], ascending = False, inplace = True)
    X_new[['user_id','Prob']].to_csv('prob_{}.csv'.format(fn), index = None)
    X_new[['user_id','pred_date']][:50000].to_csv('{}.csv'.format(fn), index = None)
    return np.mean(Performance),clf

def predict2(X_all, X_new, features, clf, score, v = False, esr=50, sk=3, fn='submission'):
    temp_user = target_order[(target_order.o_day_series < 336) & (target_order.o_day_series >= 274)][['user_id']].drop_duplicates().reset_index(drop=True)
    temp_user['CreateGroup'] = 336
    print('before delete: {}'.format(X_all.shape))
    X = temp_user.merge(X_all,on=['user_id','CreateGroup'],how = 'left')
    print('after delete: {}'.format(X.shape))

    temp_user = target_order[(target_order.o_day_series < 366) & \
                             (target_order.o_day_series >= 366 - 74)][['user_id']].drop_duplicates().reset_index(drop=True)
    temp_user['CreateGroup'] = 366
    print(-1 in temp_user.user_id)
    print(4366 in temp_user.user_id)
    print('before delete: {}'.format(X_new.shape))
    X_new = temp_user.merge(X_new,on=['user_id','CreateGroup'],how = 'left')

    temp_user = target_order[(target_order.o_day_series < 306) & (target_order.o_day_series >= 215)][['user_id']].drop_duplicates().reset_index(drop=True)
    temp_user['CreateGroup'] = 306
    print('before delete: {}'.format(X_all.shape))
    X2 = temp_user.merge(X_all,on=['user_id','CreateGroup'],how = 'left')
    print('Train: {}'.format(X_new.shape))

    kf = KFold(n_splits=sk)
    print(len(features))
    Performance = []
    X_new['Prob'] = 0
    X_new['Prob_x'] = 0
    X['Prob_x'] = 0
    for train_index, test_index in kf.split(X2):
        X_train, X_test = X2.ix[train_index,:], X2.ix[test_index,:]
        X_train, X_test = X_train[features], X_test[features]
        y_train, y_test = X2.ix[train_index,:].buy, X2.ix[test_index,:].buy
        clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='auc', verbose=v, early_stopping_rounds=esr)
        X_new['Prob_x'] = X_new['Prob_x'] + clf.predict_proba(X_new[features])[:,1]/sk
        X['Prob_x'] = X['Prob_x'] + clf.predict_proba(X[features])[:,1]/sk
    features.append('Prob_x')
   
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.ix[train_index,:], X.ix[test_index,:]
        X_train, X_test = X_train[features], X_test[features]
        y_train, y_test = X.ix[train_index,:].buy, X.ix[test_index,:].buy
        clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='auc', verbose=v, early_stopping_rounds=esr)
        pred = clf.predict_proba(X_test)[:,1]
        X_new['Prob'] = X_new['Prob'] + clf.predict_proba(X_new[features])[:,1]/sk
        Performance.append(roc_auc_score(y_test,pred))
    print("Mean Score: {}".format(np.mean(Performance)))
    importantlist = []
    for i, j in zip(features,clf.feature_importances_):
        importantlist.append([j,i])
    print(sorted(importantlist)[::-1])
    first_day = datetime.datetime.strptime('2017-08-31 00:00:00', '%Y-%m-%d %H:%M:%S')
    X_new['Days'] = np.random.randint(15,size=len(X_new))
    X_new['pred_date'] = X_new['Days'].apply(lambda x: (datetime.timedelta(days=x) + first_day).strftime("%Y-%m-%d"))
    X_new.sort_values(by = ['Prob'], ascending = False, inplace = True)
    X_new[['user_id','Prob']].to_csv('prob_{}.csv'.format(fn), index = None)
    X_new[['user_id','pred_date']][:50000].to_csv('{}.csv'.format(fn), index = None)
    return np.mean(Performance),clf

def predict(X_all, X_new, features, clf, score, v = False, esr=50, sk=3, fn='submission'):
    first_day = datetime.datetime.strptime('2017-08-31 00:00:00', '%Y-%m-%d %H:%M:%S')
    temp_user = target_order[(target_order.o_day_series < 336) & (target_order.o_day_series >= 274)][['user_id']].drop_duplicates().reset_index(drop=True)
    temp_user['CreateGroup'] = 336
    print('before delete: {}'.format(X_all.shape))
    X = temp_user.merge(X_all,on=['user_id','CreateGroup'],how = 'left')
    print('after delete: {}'.format(X.shape))
    temp_user = target_order[(target_order.o_day_series < 366) & \
                             (target_order.o_day_series >= 366 - 75)][['user_id']].drop_duplicates().reset_index(drop=True)
    temp_user['CreateGroup'] = 366
    print('before delete: {}'.format(X_new.shape))
    X_new = temp_user.merge(X_new,on=['user_id','CreateGroup'],how = 'left')
    print('Train: {}'.format(X_new.shape))
    kf = KFold(n_splits=sk)
    print(len(features))
    Performance = []
    X_new['Prob'] = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.ix[train_index,:], X.ix[test_index,:]
        X_train, X_test = X_train[features], X_test[features]
        y_train, y_test = X.ix[train_index,:].buy, X.ix[test_index,:].buy
        clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='auc', verbose=v, early_stopping_rounds=esr)
        pred = clf.predict_proba(X_test)[:,1]
        X_new['Prob'] = X_new['Prob'] + clf.predict_proba(X_new[features])[:,1]/sk
        Performance.append(roc_auc_score(y_test,pred))
    print("Mean Score: {}".format(np.mean(Performance)))
    X_new['Days'] = np.random.randint(15,size=len(X_new))
    X_new['pred_date'] = X_new['Days'].apply(lambda x: (datetime.timedelta(days=x) + first_day).strftime("%Y-%m-%d"))
    X_new.sort_values(by = ['Prob'], ascending = False, inplace = True)
    X_new[['user_id','Prob']].to_csv('prob_{}.csv'.format(fn), index = None)
    X_new[['user_id','pred_date']][:50000].to_csv('{}.csv'.format(fn), index = None)
    return np.mean(Performance),clf

def validateseq2(X_all, y, features, clf, score, v = False, esr=50, sk=5):
    temp_user = target_order[(target_order.o_day_series < 336) & (target_order.o_day_series >= 274)][['user_id']].drop_duplicates().reset_index(drop=True)
    temp_user['CreateGroup'] = 336
    print('before delete: {}'.format(X_all.shape))
    X = temp_user.merge(X_all,on=['user_id','CreateGroup'],how = 'left')
    print('after delete: {}'.format(X.shape))
    temp_user = target_order[(target_order.o_day_series < 306) & (target_order.o_day_series >= 215)][['user_id']].drop_duplicates().reset_index(drop=True)
    temp_user['CreateGroup'] = 306
    print('before delete: {}'.format(X_all.shape))
    X2 = temp_user.merge(X_all,on=['user_id','CreateGroup'],how = 'left')
    print('after delete: {}'.format(X.shape))
    kf = KFold(n_splits=sk)
    print(len(features))
    X['Prob_x'] = 0
    for train_index, test_index in kf.split(X2):
        X_train, X_test = X2.ix[train_index,:], X2.ix[test_index,:]
        X_train, X_test = X_train[features], X_test[features]
        y_train, y_test = X2.ix[train_index,:].buy, X2.ix[test_index,:].buy
        clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='auc', verbose=v, early_stopping_rounds=esr)
        X['Prob_x'] = X['Prob_x'] + clf.predict_proba(X[features])[:,1]/sk
    Performance = []
    features.append('Prob_x')
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.ix[train_index,:], X.ix[test_index,:]
        X_train, X_test = X_train[features], X_test[features]
        y_train, y_test = X.ix[train_index,:].buy, X.ix[test_index,:].buy
        clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='auc', verbose=v, early_stopping_rounds=esr)
        pred = clf.predict_proba(X_test)[:,1]
        Performance.append(roc_auc_score(y_test,pred))
    print("Mean Score: {}".format(np.mean(Performance)))
    return np.mean(Performance),clf

def read6810():
    mergeindex = ['user_id','CreateGroup']
    mergefeatures = []
#    train = pd.read_csv('traina_8month_more.csv')
    key = ['_o_area','_sku_id']
    mainfn = 'trainb_8month_final.csv'
    mergefn = ['trainb_1month_final.csv']
    train = Merge(mainfn, mergefn, key)
    train_other = pd.read_csv('trainb_8month_other_final.csv')
    for i in train_other.columns:
        if ('cnt' in i) | ('median' in i) | ('Difference' in i) | ('sum' in i) | ('Plux' in i) | ('series' in i):
            mergefeatures.append(i)
    train_other = train_other[mergeindex + mergefeatures]
    mergefeatures2 = [i+'_other' for i in mergefeatures]
    train_other.columns = mergeindex + mergefeatures2
    train = pd.merge(train, train_other, on = mergeindex, how = 'left')
#    pre = pd.read_csv('testa_8month_more.csv')
    print(np.unique(train.CreateGroup))
    mainfn = 'testb_8month_final.csv'
    mergefn = ['testb_1month_final.csv']
    pre = Merge(mainfn, mergefn, key)
    pre_other = pd.read_csv('testb_8month_other_final.csv')
    pre_other = pre_other[mergeindex + mergefeatures]
    pre_other.columns = mergeindex + mergefeatures2
    pre = pd.merge(pre, pre_other, on = mergeindex, how = 'left')
    print(np.unique(pre.CreateGroup))
    df = pd.concat([train, pre]).reset_index(drop=True)
    df2 = pd.read_csv('D2b.csv')
    df2_other = pd.read_csv('D2b_other.csv')
    print(np.unique(df2.CreateGroup))
    cold = list(df2.columns)
    cold.remove('user_id')
    cold.remove('CreateGroup')
    df2_other = df2_other[['user_id','CreateGroup'] + cold]
    for i in range(len(cold)):
        cold[i] = cold[i] + '_other'
    df2_other.columns = ['user_id','CreateGroup'] + cold
    df2 = df2.merge(df2_other,on = ['user_id','CreateGroup'])
    col2 = list(df2.columns)
    for i in col2[:]:
        if ('price' in i) | ('para' in i) | ('o_month' in i) | ('o_date' in i)| ('age' in i)| ('sex' in i) | ('user_lv_cd' in i):
            col2.remove(i)
    df = df.merge(df2[col2],on=['user_id','CreateGroup'],how = 'left')
    df = df.merge(order_hist([246,276,306,336,366],5,'o_sku_num') ,on=['user_id','CreateGroup'],how = 'left')
    df = df.merge(order_hist([246,276,306,336,366],5,'o_day_series') ,on=['user_id','CreateGroup'],how = 'left')
    df = df.merge(order_hist([246,276,306,336,366],5,'para_1') ,on=['user_id','CreateGroup'],how = 'left')
    df = df.merge(order_price_total_hist([246,276,306,336,366],5),on=['user_id','CreateGroup'],how = 'left')
    col = list(df.columns)
    last = pd.read_csv('last3b.csv')
    print(np.unique(last.CreateGroup))
    df = df.merge(last,on = ['user_id','CreateGroup'],how = 'left')
#    print(df[['user_id','CreateGroup','nextbuy_1','nextbuy_2','nextbuy_3','nextbuy_4']])
    print(df.shape)
    return df

def validate(X_all, y, features, clf, score, v = False, esr=50, sk=5):
    temp_user = target_order[(target_order.o_day_series < 336) & (target_order.o_day_series >= 273)][['user_id']].drop_duplicates().reset_index(drop=True)
    temp_user['CreateGroup'] = 336
    print('before delete: {}'.format(X_all.shape))
    X = temp_user.merge(X_all,on=['user_id','CreateGroup'],how = 'left')
    print('after delete: {}'.format(X.shape))
    kf = KFold(n_splits=sk)
    print(len(features))
    Performance = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.ix[train_index,:], X.ix[test_index,:]
        X_train, X_test = X_train[features], X_test[features]
        y_train, y_test = X.ix[train_index,:].buy, X.ix[test_index,:].buy
        clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='auc', verbose=v, early_stopping_rounds=esr)
        pred = clf.predict_proba(X_test)[:,1]
        Performance.append(roc_auc_score(y_test,pred))
    print("Mean Score: {}".format(np.mean(Performance)))
    return np.mean(Performance),clf

df = read6810()
df.ix[df.CreateGroup == 367, 'CreateGroup'] = 366
df = df.drop_duplicates(subset=['user_id','CreateGroup'],keep='last').reset_index(drop = True)
print(df.shape)
t3 = pd.read_csv('trainb_3month_3level_userid_2.csv')
t3p = pd.read_csv('testb_3month_3level_userid_2.csv')
t3 = pd.concat([t3,t3p])
notusable = ['buy','nextbuy','o_date','a_date','PredictDays','user_id','CreateGroup']
month = [i for i in t3.columns if i not in notusable]
tt = t3[['user_id','CreateGroup'] + month]
tt.columns = ['user_id','CreateGroup'] + ['{}_3month'.format(i) for i in month]
df2 = df.merge(tt, on = ['user_id','CreateGroup'], how = 'left')
df3 = df2[:]
for i in ['o_day_series_o_ave','o_day_series_o_sum','o_day_series_o_median','o_day_series_o_min','o_day_series_o_max','o_day_series',
         'c_day_series_o_ave','c_day_series_o_sum','c_day_series_o_median','c_day_series_o_min','c_day_series_o_max','c_day_series',
         'a_day_series_a_ave','a_day_series_a_sum','a_day_series_a_median','a_day_series_a_min','a_day_series_a_max','a_day_series',
         'o_day_series_0','o_day_series_1','o_day_series_2','o_day_series_3','o_day_series_4',
         'o_day_series_o_ave_o_user_id_3month','o_day_series_o_sum_o_user_id_3month','o_day_series_o_min_o_user_id_3month',
         'o_day_series_o_max_o_user_id_3month','o_day_series_3month','c_day_series_o_ave_o_user_id_3month','c_day_series_o_sum_o_user_id_3month',
         'c_day_series_o_min_o_user_id_3month','c_day_series_o_max_o_user_id_3month','c_day_series_3month',
         'a_day_series_a_ave_3month','a_day_series_a_sum_3month','a_day_series_a_min_3month','a_day_series_a_max_3month','a_day_series_3month',
         'o_day_series_o_median_other','c_day_series_o_median_other','a_day_series_a_median_other','o_day_series_other',
         'a_day_series_other','c_day_series_other']:
    df3[i] = df3['CreateGroup'] - df3[i]

for i in ['o_month_series_o_ave','o_month_series_o_sum','o_month_series_o_median','o_month_series_o_min','o_month_series_o_max','o_month_series',
         'a_month_series_a_ave','a_month_series_a_sum','a_month_series_a_median','a_month_series_a_min','a_month_series_a_max','a_month_series',
         'o_month_series_o_median_other','a_month_series_a_median_other',
         'o_month_series_o_ave_o_user_id_3month','o_month_series_o_sum_o_user_id_3month','o_month_series_o_min_o_user_id_3month',
         'o_month_series_o_max_o_user_id_3month','o_month_series_3month',
         'a_month_series_a_ave_3month','a_month_series_a_sum_3month','a_month_series_a_min_3month','a_month_series_a_max_3month']:
    df3[i] = df3['CreateGroup']//30 - df3[i]

clf = lgbm.LGBMClassifier(objective='binary', num_leaves=35, max_depth=-1,
                          learning_rate=0.02, seed=1, colsample_bytree=0.8, subsample=0.8, n_estimators=10000)

fa = readlog('record_imp_auc2.log',0.765255)
for i in fa[:]:
    if ('series_o_sum' in i) | ('series_a_sum' in i):
        fa.remove(i)
'''
different = ['price_o_ave_o_cate_o_ave_o_user_id_3month','price_o_ave_o_cate_o_min_o_user_id_3month',
            'price_o_ave_o_cate_o_max_o_user_id_3month','price_o_ave_o_cate_3month','para_1_o_ave_o_cate_o_ave_o_user_id_3month',
            'para_1_o_ave_o_cate_o_min_o_user_id_3month','para_1_o_ave_o_cate_o_max_o_user_id_3month',
            'para_1_o_ave_o_cate_3month','para_2_o_ave_o_cate_o_ave_o_user_id_3month',
            'para_2_o_ave_o_cate_o_min_o_user_id_3month','para_2_o_ave_o_cate_o_max_o_user_id_3month',
            'para_2_o_ave_o_cate_3month','para_3_o_ave_o_cate_o_ave_o_user_id_3month',
            'para_3_o_ave_o_cate_o_min_o_user_id_3month','para_3_o_ave_o_cate_o_max_o_user_id_3month',
            'para_3_o_ave_o_cate_3month','price_a_ave_cate_a_min_3month','price_a_ave_cate_a_max_3month',
            'price_a_ave_cate_3month','para_1_a_ave_cate_a_min_3month','para_1_a_ave_cate_a_max_3month',
            'para_1_a_ave_cate_3month',]

fa = readlog('record_imp_auc2.log',0.765255)
fa = readlog('record_imp_auc2.log',0.765472)
#fa = readlog('record_imp_auc2.log',0.765895)
#fa = fa + [j for j in ['{}_3month'.format(i) for i in month] if (j not in fa) & ('Plux' in j)]
for i in fa[:]:
    if ('series_o_sum' in i) | ('series_a_sum' in i)| ('c_day' in i) | (i in different):
        fa.remove(i)
'''
#fa = [i for i in df3.columns if i not in notusable]
#_,t = validateseq2(df3, df3, fa, clf, score,v = False, sk=5)
#_,t = validate(df3, df3, fa, clf, score, sk=5, v = False)
predict22(df3[df3.CreateGroup < 337].reset_index(), df3[df3.CreateGroup > 337].reset_index(), 
        fa, clf, score, fn='submission_7627_5216', sk=5, v = False)

