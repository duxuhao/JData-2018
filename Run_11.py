from History_detailB import order_hist,order_price_total_hist,action_hist
from utility import Merge,readlog
import xgboost as xgb
import lightgbm as lgbm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

order = pd.read_csv('./B/jdata_user_order.csv', parse_dates=['o_date'])
sku = pd.read_csv('./B/jdata_sku_basic_info.csv', )
order = pd.merge(order, sku, on='sku_id', how='left')
target_order = order[(order.cate == 101) | (order.cate == 30)].reset_index(drop=True)
first_day = datetime.datetime.strptime('2016-08-31 00:00:00', '%Y-%m-%d %H:%M:%S')
target_order['o_day_series'] = (target_order['o_date'] - first_day).apply(lambda x: x.days)
order['o_day_series'] = (order['o_date'] - first_day).apply(lambda x: x.days)

def predictsecond(Xtrain_all, Xpre, features, clf,fn):
    first_day = datetime.datetime.strptime('2017-05-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    all = []
    day = 336
    for i in range(4):
        temp_user = target_order[(target_order.o_day_series < day-30*i) & (target_order.o_day_series >= day - 90-30*i)][['user_id']].drop_duplicates().reset_index(drop=True)
        temp_user['CreateGroup'] = day-30*i
        all.append(temp_user)
    temp_user = pd.concat(all).reset_index(drop=True)
    print('before delete: {}'.format(Xtrain_all.shape))
    Xtrain = temp_user.merge(Xtrain_all,on=['user_id','CreateGroup'],how = 'left')
    Xtrain = Xtrain[Xtrain.CreateGroup > Xtrain.CreateGroup.min()].reset_index(drop = True)
    print('Train: {}'.format(Xtrain.shape))
    clf[0].fit(Xtrain[features[0]],Xtrain.buy, eval_set = [(Xtrain[features[0]],Xtrain.buy)], eval_metric='auc', verbose=True)
    Xpre['Prob_x'] = clf[0].predict_proba(Xpre[features[0]])[:,1]
    Xtrain['Prob_x'] = clf[0].predict_proba(Xtrain[features[0]])[:,1]
    features[1].append('Prob_x')
    Xtrain.fillna(0,inplace = True)
    Xpre.fillna(0,inplace = True)
    clf[1].fit(Xtrain[features[1]],Xtrain.buy, eval_set = [(Xtrain[features[1]],Xtrain.buy)], verbose=True)
    Xpre['Prob'] = clf[1].predict(Xpre[features[1]])
    #clf[1].fit(Xtrain_all[features[1]],Xtrain_all.nextbuy, eval_set = [(Xtrain_all[features[1]],Xtrain_all.nextbuy)], verbose=True)
    #Xpre['Days'] = clf[1].predict(Xpre[features[1]])
    Xpre['Days'] = 7
    Xpre['pred_date'] = Xpre['Days'].apply(lambda x: (datetime.timedelta(days=x) + first_day).strftime("%Y-%m-%d"))
    Xpre.sort_values(by = ['Prob'], ascending = False, inplace = True)
    Xpre[['user_id','Prob']].to_csv('prob_{}.csv'.format(fn), index = None)
    Xpre[['user_id','pred_date']][:50000].to_csv('{}.csv'.format(fn), index = None)
    return Xpre

def score(pred, real): #评分系统，感谢herhert，只要s1
    pred['index'] = np.arange(pred.shape[0]) + 1
    pred['wi'] = 1 / (1 + np.log(pred['index']))

    compare = pd.merge(pred, real, how='left', on='user_id')
    compare = compare[:40000]
    compare.fillna(0, inplace=True)
    S10 = np.sum(compare['buy'] * compare['wi']) / compare['wi'].sum()
    print('40000: {}'.format(S10))
    compare = compare[:30000]
    compare.fillna(0, inplace=True)
    S10 = np.sum(compare['buy'] * compare['wi']) / compare['wi'].sum()
    print('30000: {}'.format(S10))
    compare = compare[:29000]
    S1 = np.sum(compare['buy'] * compare['wi']) / compare['wi'].sum()
    print('29000: {}'.format(S1))
    compare = compare[:28000]
    S1 = np.sum(compare['buy'] * compare['wi']) / compare['wi'].sum()
    print('28000: {}'.format(S1))
    return S10

def validate2(X_all, y, features, clf, score, v = False, esr=50):
    for day in [336]:
        all = []
        for i in range(4):
            if i == 0:
                temp_user = target_order[(target_order.o_day_series < day-30*i) & (target_order.o_day_series >= day - 90-30*i)][['user_id']].drop_duplicates().reset_index(drop=True)
            else:
                temp_user = target_order[(target_order.o_day_series < day-30*i) & (target_order.o_day_series >= day - 90-30*i)][['user_id']].drop_duplicates().reset_index(drop=True)
            temp_user['CreateGroup'] = day-30*i
#            print(temp_user.shape)
            all.append(temp_user)
        temp_user = pd.concat(all).reset_index(drop=True)
        print('before delete: {}'.format(X_all.shape))
        X = temp_user.merge(X_all,on=['user_id','CreateGroup'],how = 'left')
        print('after delete: {}'.format(X.shape))
        Ttrain = X.CreateGroup < day
        Ttest = (X.CreateGroup == day)
        Testtemp = X[Ttest]
        X_train, X_test = X[Ttrain], X[Ttest] 
        y_train, y_test = X_train.buy, X[Ttest].buy
        X_train, X_test = X_train[features[0]], X_test[features[0]]
        print('Train: {}'.format(X_train.shape))
        print('Test: {}'.format(X_test.shape))
        clf[0].fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='auc', verbose=v, early_stopping_rounds=esr)
        X['Prob_x'] = clf[0].predict_proba(X[features[0]])[:,1]
        Testtemp['Prob_x'] = clf[0].predict_proba(X_test)[:,1]
        X_train, X_test = X[Ttrain], X[Ttest]
        features[1].append('Prob_x')
        X_train, X_test = X_train[features[1]], X_test[features[1]]
        X_train.fillna(0,inplace = True)
        X_test.fillna(0,inplace = True)
        y_train.fillna(0,inplace = True)
        y_test.fillna(0,inplace = True)
        clf[1].fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)],  verbose=v, early_stopping_rounds=esr)
        print(X_test.shape)
        print(X[Ttest].buy.sum())
        Testtemp['Prob'] = clf[1].predict(X_test)
        Testtemp['Days'] = 7
        prediction = Testtemp[['user_id','Prob','Days']]
        prediction.sort_values(by = ['Prob'], ascending = False, inplace = True)
        prediction.to_csv('validate.csv',index = None)
        Performance = score(prediction[['user_id','Days']], Testtemp[['user_id', 'nextbuy','buy']])
    print("Mean Score: {}".format(Performance))
    return Performance,clf, Testtemp

def read6810():
    mergeindex = ['user_id','CreateGroup']
    mergefeatures = []
    key = ['_o_area','_sku_id']
    mainfn = 'trainb_8month_final.csv'
    mergefn = ['trainb_1month_final.csv']
    train = Merge(mainfn, mergefn, key)
    train_other = pd.read_csv('trainb_8month_other_final.csv')
    for i in train_other.columns:
        if ('cnt' in i) | ('median' in i) | ('Difference' in i) | ('o_sku_num_o_sum' in i) | ('a_num_a_sum' in i):
            mergefeatures.append(i)
    train_other = train_other[mergeindex + mergefeatures]
    mergefeatures2 = [i+'_other' for i in mergefeatures]
    train_other.columns = mergeindex + mergefeatures2
    train = pd.merge(train, train_other, on = mergeindex, how = 'left')
#    print('Final')
#    print(train.shape)
#    print(train.drop_duplicates(subset = ['user_id','CreateGroup']).shape)
    mainfn = 'testb_8month_final.csv'
    mergefn = ['testb_1month_final.csv']
    pre = Merge(mainfn, mergefn, key)
    pre_other = pd.read_csv('testb_8month_other_final.csv')
    pre_other = pre_other[mergeindex + mergefeatures]
    pre_other.columns = mergeindex + mergefeatures2
    pre = pd.merge(pre, pre_other, on = mergeindex, how = 'left')
    df = pd.concat([train, pre]).reset_index(drop=True)
#    print(df.shape)
#    print(df.drop_duplicates(subset = ['user_id','CreateGroup']).shape)
    df2 = pd.read_csv('D2b.csv')
#    print('D')
#    print(df2.shape)
#    print(df2.drop_duplicates(subset = ['user_id','CreateGroup']).shape)
    df2_other = pd.read_csv('D2b_other.csv')
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
    df = df.merge(last,on = ['user_id','CreateGroup'],how = 'left')
    return df

df = read6810()

#print(df.shape)
df = df.drop_duplicates(subset=['user_id','CreateGroup'],keep='last').reset_index(drop = True)
#print(df.shape)

t3 = pd.read_csv('trainb_3month_3level_userid_2.csv')
t3p = pd.read_csv('testb_3month_3level_userid_2.csv')
t3 = pd.concat([t3,t3p])

notusable = ['buy','nextbuy','o_date','a_date','PredictDays','user_id','CreateGroup']
month = [i for i in t3.columns if i not in notusable]
tt = t3[['user_id','CreateGroup'] + month]
tt.columns = ['user_id','CreateGroup'] + ['{}_3month'.format(i) for i in month]
df2 = df.merge(tt, on = ['user_id','CreateGroup'], how = 'left')

clf = lgbm.LGBMClassifier(objective='binary', num_leaves=35, max_depth=-1,
                          learning_rate=0.05, seed=1, colsample_bytree=0.8, subsample=0.8, n_estimators=155)

clf2 = lgbm.LGBMRegressor(num_leaves=13, max_depth=4,learning_rate=0.05, seed=1, colsample_bytree=0.8, subsample=0.8, n_estimators=98)
fa = readlog('record_seq_3month.log',0.683594)

fa2 = ['nextbuy_1','nextbuy_2','nextbuy_3','nextbuy_4','nextbuy_5','nextbuy_6','nextbuy_7','nextbuy_8','daybeforelastbuy_o_sum','o_day_series','CommentEndDateDifference']

#Res = validate2(df2, df2, [fa,fa2], [clf,clf2], score,v = True)
predictsecond(df2[df2.CreateGroup < 337].reset_index(), df2[df2.CreateGroup > 337].reset_index(), [fa,fa2],[clf,clf2],'submissionb_618_5200')
