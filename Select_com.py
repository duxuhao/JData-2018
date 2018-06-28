import lightgbm as lgbm
import pandas as pd
import numpy as np
from MLFeatureSelection import sequence_selection, importance_selection, coherence_selection,tools


def score2(pred, real): #针对s2的评分函数
    print('score2')
    compare = pd.merge(pred, real, how='left', on='user_id')
    compare_for_S2 = compare[compare['buy'] == 1]
    S2 = np.sum(10 / (10 + np.square(compare_for_S2['Days'] - compare_for_S2['nextbuy']))) / compare_for_S2.shape[0]
    return S2

def score1(pred, real): #针对s1的评分函数
    pred['index'] = np.arange(pred.shape[0]) + 1
    pred['wi'] = 1 / (1 + np.log(pred['index']))

    compare = pd.merge(pred, real, how='left', on='user_id')
    compare = compare[:34000]
    compare.fillna(0, inplace=True)
    S1 = np.sum(compare['buy'] * compare['wi']) / compare['wi'].sum()
    return S1

def read1():
    df = pd.read_csv('train.csv')
    return df

def validate_clf(X, y, features, clf, score, v = False): #用于挑选分类器特征
    day = 335 #这里你可以换成月份，反正更换Ttrain和Ttest来表征你的训练集和验证集就行
    Ttrain = X.CreateGroup < day
    Ttest = (X.CreateGroup == day)
    Testtemp = X[Ttest]
    X_train, X_test = X[X.buy==1][Ttrain], X[X.buy==1][Ttest]
    X_train, X_test = X_train[features], X_test[features]
    print('Train: {}'.format(X_train.shape))
    y_train, y_test = X[X.buy==1][Ttrain].nextbuy, X[X.buy==1][Ttest].nextbuy
    print('train')
    clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric = 'auc', verbose=v, early_stopping_rounds=50)
    Testtemp['Prob'] = clf.predict_proba(X_test)[:,1]
    Testtemp['Days'] = 7
    print(Testtemp['Days'])
    prediction = Testtemp[['user_id','Days','Prob']]
    prediction.sort_values(by = ['Prob'], ascending = False, inplace = True)
    Performance = score(prediction[['user_id','Days']], Testtemp[['user_id', 'nextbuy','buy']])
    print("Mean Score: {}".format(Performance))
    return Performance, clf

def validate_reg(X, y, features, clf, score, v = False): #用于挑选回归器特征
    day = 335 #这里你可以换成月份，反正更换Ttrain和Ttest来表征你的训练集和验证集就行
    Ttrain = X.CreateGroup < day
    Ttest = (X.CreateGroup == day)
    Testtemp = X[Ttest]
    X_train, X_test = X[X.buy==1][Ttrain], X[X.buy==1][Ttest]
    X_train, X_test = X_train[features], X_test[features]
    print('Train: {}'.format(X_train.shape))
    y_train, y_test = X[X.buy==1][Ttrain].nextbuy, X[X.buy==1][Ttest].nextbuy
    print('train')
    clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], verbose=v, early_stopping_rounds=50)
    Testtemp['Days'] = clf.predict(Testtemp[features])
    print(Testtemp['Days'])
    prediction = Testtemp[['user_id','Days']]
    Performance = score(prediction[['user_id','Days']], Testtemp[['user_id', 'nextbuy','buy']])
    print("Mean Score: {}".format(Performance))
    return Performance, clf

def seq(df,f, notusable,clf): #序列搜索
    sf = sequence_selection.Select(Sequence = True, Random = True, Cross = True) #初始化选择器，选择你需要的流程
    sf.ImportDF(df,label = 'nextbuy') #导入数据集以及目标标签
    sf.ImportLossFunction(score2, direction = 'ascend') #导入评价函数以及优化方向，所是回归则导入score2，若是分类则是score1
    sf.ImportCrossMethod(CrossMethod)
    sf.InitialNonTrainableFeatures(notusable) #初始化不能用的特征
    sf.InitialFeatures(f)
    sf.GenerateCol() #生成特征库 （具体该函数变量请参考根目录下的readme）
#    sf.SetTimeLimit(120) #设置算法运行最长时间，以分钟为单位
    sf.clf = clf
    sf.SetLogFile('record_seq_reg.log') #初始化日志文件
    return sf.run(validate_reg) #输入检验函数并开始运行，回归则导入validate_reg，若是分类则是validate_clf

def imp(df,f,clf): #根据重要性删减
    sf = importance_selection.Select() #初始化选择器，选择你需要的流程
    sf.ImportDF(df,label = 'nextbuy') #导入数据集以及目标标签
    sf.ImportLossFunction(score2, direction = 'ascend') #导入评价函数以及优化方向
    sf.InitialFeatures(f)
    sf.SelectRemoveMode(frac = 0.05)
    sf.clf = clf
    sf.SetLogFile('record_imp_reg.log')
    return sf.run(validate_reg) #回归则导入validate_reg，若是分类则是validate_clf

def coh(df,f,clf): #根据相关性删减
    sf = coherence_selection.Select() #初始化选择器，选择你需要的流程
    sf.ImportDF(df,label = 'nextbuy') #导入数据集以及目标标签
    sf.ImportLossFunction(score2, direction = 'ascend') #导入评价函数以及优化方向
    sf.InitialFeatures(f)
    sf.SelectRemoveMode(frac=0.05, lowerbound = 0.9)
    sf.clf = clf
    sf.SetLogFile('record_coh_reg.log') #初始化日志文件
    return sf.run(validate_reg) #回归则导入validate_reg，若是分类则是validate_clf

if __name__ == "__main__":
    df = read1()

    notusable = ['buy','nextbuy','o_date','a_date','PredictDays','user_id'] #自己定义无法进行训练的特征
    f = list(df.columns)
    for i in notusable:
        f.remove(i)

    clf = lgbm.LGBMClassifier(objective='binary', num_leaves=35, max_depth=-1, learning_rate=0.05, seed=1, colsample_bytree=0.8, subsample=0.8, n_estimators=1000)
    est = lgbm.LGBMRegressor(random_state=1, num_leaves =6, n_estimators=1000, max_depth=3, learning_rate = 0.05, colsample_bytree=0.8, subsample=0.8)
#    f = tools.readlog('record_seq_reg.log',0.15705) #读取之前最好的结果
#    f = [] #若你有起始的特征组合，直接以list的形式放进来，跑过一次之后就可以通过tools.readlog()读进来了
    n = 1
    uf = f[:]
    while n | (uf != f):
        n = 0
        print('importance selection')
        uf = imp(df,uf,est) #先做重要性搜索
        print('coherence selection')
        uf = coh(df,uf,est) #然后将最好结果传递下去，然后进行相关性搜索
        print('sequence selection')
        seq(df, uf, notusable,est) #最后进行序列搜索
