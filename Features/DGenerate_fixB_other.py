# question2 2018-05-13
import pandas as pd
from datetime import timedelta

print('loading data...')
sku_basic = pd.read_csv('../B/jdata_sku_basic_info.csv')
user_basic = pd.read_csv('../B/jdata_user_basic_info.csv')
user_action = pd.read_csv('../B/jdata_user_action.csv')
user_order = pd.read_csv('../B/jdata_user_order.csv')
user_comment = pd.read_csv('../B/jdata_user_comment_score.csv')


print('getting datetime...')
user_order['o_date'] = pd.to_datetime(user_order['o_date'])
user_order['o_month'] = user_order['o_date'].dt.month
user_order['o_day'] = user_order['o_date'].dt.day
user_action['a_date'] = pd.to_datetime(user_action['a_date'])
user_action['a_month'] = user_action['a_date'].dt.month
user_comment['comment_create_tm'] = pd.to_datetime(user_comment['comment_create_tm'])
user_comment['comment_month'] = user_comment['comment_create_tm'].dt.month

print('dropping duplicates...')
user_order = user_order.drop_duplicates()
user_order = user_order.groupby(['user_id', 'sku_id', 'o_id', 'o_date', 'o_month', 'o_day', 'o_area'])['o_sku_num'].mean().reset_index()
user_comment = user_comment.sort_values(by=['user_id', 'o_id', 'comment_create_tm'])
user_comment = user_comment.groupby(['user_id', 'o_id']).tail(1)
user_comment['comment_create_tm'] = user_comment['comment_create_tm'].apply(lambda x: str(x)[:10])
user_comment['comment_create_tm'] = pd.to_datetime(user_comment['comment_create_tm'])

#20180424新增删除浏览次数过多的(0513 没有作用)
user_action = user_action.loc[user_action['a_num'] <= 20]

#生成带标签类的数据
print('getting label data...')
user_order_cate = user_order.merge(sku_basic[['sku_id', 'cate']], on='sku_id', how='left')
user_order_spec1 = user_order_cate.loc[user_order_cate['cate'] == 101]
user_order_spec2 = user_order_cate.loc[user_order_cate['cate'] == 30]
user_order_spec = pd.concat([user_order_spec1, user_order_spec2])
user_order_label_temp = user_order_spec[['user_id', 'o_month', 'o_day','o_date']].drop_duplicates()

user_order_label_temp = user_order_label_temp.sort_values(by=['user_id', 'o_month', 'o_day'])
user_order_label = user_order_label_temp.groupby(['user_id', 'o_month']).head(1)
order_label = user_order_label.copy()

print('getting feature table...')
#用户订单表特征表
user_order_table = user_order.copy()
#用户订单-sku特征表
sku_basic['para_1'] = sku_basic['para_1'].replace(-1, sku_basic['para_1'].mean())
order_sku_table = user_order.merge(sku_basic, on='sku_id', how='left')
order_sku_table['money'] = order_sku_table['o_sku_num'] * order_sku_table['price']
#用户行为特征表
user_action_table = user_action.copy()
#用户行为-sku特征表
action_sku_table = user_action.merge(sku_basic, on='sku_id', how='left')
#评分特征表
comment_table = user_comment.copy()
#订单-评分特征表
order_comment = user_order.merge(user_comment, on=['user_id', 'o_id'], how='left')
order_comment = order_comment.fillna(-1)
order_comment_level = order_comment.loc[order_comment['score_level'] != -1]
#订单-sku-评分表
order_sku_comment = user_order.merge(sku_basic, on='sku_id', how='left').\
    merge(user_comment, on=['user_id', 'o_id'], how='left')
order_sku_comment = order_sku_comment.fillna(-1)
sku_comment = order_sku_comment.loc[order_sku_comment['score_level'] != -1]


def past_order_process(start_date, end_date, a_day):
    print('processing past order...')
    order_day_diff1 = user_order_table.loc[user_order_table['o_date'] < end_date]
    order_day_diff2 = order_day_diff1.loc[order_day_diff1['o_date'] >= start_date]

    order_day_diff2['past_day'] = order_day_diff2['o_date'].apply(lambda x: (end_date - x).days)
    order_day_diff2 = order_day_diff2.sort_values(by=['user_id', 'past_day'])

    ############################0424
    order_date = order_day_diff2[['user_id', 'o_date']].drop_duplicates()
    order_date = order_date.sort_values(by=['user_id', 'o_date'])
    order_date['date_shift'] = order_date.groupby(['user_id'])['o_date'].shift(-1)
    fill_date = pd.to_datetime('1900-01-01')
    order_date['date_shift'] = order_date['date_shift'].fillna(fill_date)
    order_date = order_date.loc[order_date['date_shift'] != fill_date]
    order_date['day_diff'] = order_date['date_shift'] - order_date['o_date']
    order_date['day_diff'] = order_date['day_diff'].apply(lambda x: x.days)
    order_date_diff = order_date.groupby(['user_id'])['day_diff'].mean().reset_index()
    order_date_diff.columns = ['user_id', 'day_diff_mean_%d' % a_day]
    ###############################0424

    user_order_near_temp = order_day_diff2.groupby('user_id').head(1)
    user_order_near = user_order_near_temp[['user_id', 'past_day']].copy()

    ###################0424
    user_order_count_near = user_order_near.merge(order_day_diff2[['user_id', 'o_sku_num', 'past_day']], \
                                                  on=['user_id', 'past_day'], how='left')
    user_order_sku_num = user_order_count_near.groupby(['user_id'])['o_sku_num'].sum().reset_index()
    user_order_sku_num.columns = ['user_id', 'near_sku_num_%d' % a_day]
    ########################0424



    user_order_near.columns = ['user_id', 'order_near_day_%d' % a_day]
    order_count_temp = order_day_diff2.groupby(['user_id', 'o_id'])['sku_id'].count().reset_index()
    order_count = order_count_temp.groupby(['user_id'])['o_id'].count().reset_index()
    order_count.columns = ['user_id', 'order_count_%d' % a_day]
    order_area_num_temp = order_day_diff2.groupby(['user_id', 'o_area'])['o_date'].count().reset_index()
    order_area_num = order_area_num_temp.groupby(['user_id'])['o_area'].count().reset_index()
    order_num = order_day_diff2.groupby(['user_id'])['o_sku_num'].sum().reset_index()
    order_num.columns = ['user_id', 'order_num_%d' % a_day]
    order_area_num.columns = ['user_id', 'order_area_num_%d' % a_day]
    order_day_temp = order_day_diff2.groupby(['user_id', 'o_date'])['o_area'].count().reset_index()
    order_day = order_day_temp.groupby(['user_id'])['o_date'].count().reset_index()
    order_day.columns = ['user_id', 'order_day_%d' % a_day]
    order_sku_temp = order_day_diff2.groupby(['user_id', 'sku_id'])['o_area'].count().reset_index()
    order_sku = order_sku_temp.groupby(['user_id'])['sku_id'].count().reset_index()
    order_sku.columns = ['user_id', 'order_sku_count_%d' % a_day]

    order_feature = user_order_near.merge(order_count, on='user_id', how='left'). \
        merge(order_num, on='user_id', how='left'). \
        merge(order_area_num, on='user_id', how='left'). \
        merge(order_day, on='user_id', how='left'). \
        merge(order_sku, on='user_id', how='left'). \
        merge(order_date_diff, on='user_id', how='left'). \
        merge(user_order_sku_num, on='user_id', how='left')
    return order_feature




def past_order_sku(start_date, end_date, a_day):
    print('processing order_sku data...')
    order_sku_day_diff1 = order_sku_table.loc[order_sku_table['o_date'] < end_date]
    order_sku_day_diff2 = order_sku_day_diff1.loc[order_sku_day_diff1['o_date'] >= start_date]
    order_sku_day_diff2['past_day'] = order_sku_day_diff2['o_date'].apply(lambda x: (end_date - x).days)

    user_price = order_sku_day_diff2.groupby(['user_id'])['price'].agg(['mean', 'max', 'min']).reset_index()
    user_price.columns = ['user_id', 'price_mean_%d' % a_day, 'price_max_%d' % a_day, 'price_min_%d' % a_day]
    user_para = order_sku_day_diff2.groupby(['user_id'])['para_1'].agg(['mean', 'max', 'min']).reset_index()
    user_para.columns = ['user_id', 'para1_mean_%d' % a_day, 'para1_max_%d' % a_day, 'para1_min_%d' % a_day]
    user_money = order_sku_day_diff2.groupby(['user_id'])['money'].sum().reset_index()
    user_money.columns = ['user_id', 'money_sum_%d' % a_day]

    cho_order_sku_C_1 = order_sku_day_diff2.loc[order_sku_day_diff2['cate'] == 101]
    cho_order_sku_C_2 = order_sku_day_diff2.loc[order_sku_day_diff2['cate'] == 30]
    cho_order_sku_C = pd.concat([cho_order_sku_C_1, cho_order_sku_C_2])
    cho_order_sku_C = cho_order_sku_C.sort_values(by=['user_id', 'past_day'])
    order_sku_C_near_temp = cho_order_sku_C.groupby(['user_id']).head(1)
    order_sku_C_near = order_sku_C_near_temp[['user_id', 'past_day']].copy()
    sku_C_near_num_temp = order_sku_C_near.merge(cho_order_sku_C[['user_id', 'past_day', 'o_sku_num', 'money']],
                                                 on=['user_id', 'past_day'], how='left')
    sku_C_near_num = sku_C_near_num_temp.groupby(['user_id', 'past_day'])['o_sku_num'].sum().reset_index()
    sku_C_near_num.columns = ['user_id', 'past_day', 'cate_sku_near_num_%d' % a_day]
    sku_C_near_num = sku_C_near_num.drop(['past_day'], axis=1)
    sku_C_near_money = sku_C_near_num_temp.groupby(['user_id', 'past_day'])['money'].sum().reset_index()
    sku_C_near_money.columns = ['user_id', 'past_day', 'cate_sku_money_%d' % a_day]
    sku_C_near_money = sku_C_near_money.drop(['past_day'], axis=1)
    order_sku_C_near.columns = ['user_id', 'cate_past_day_%d' % a_day]

    ####################0424
    order_sku_C_day_temp = cho_order_sku_C.groupby(['user_id', 'o_date'])['o_area'].count().reset_index()
    order_sku_C_day = order_sku_C_day_temp.groupby(['user_id'])['o_date'].count().reset_index()
    order_sku_C_day.columns = ['user_id', 'order_sku_day_%d' % a_day]
    ####################


    order_sku_C_count_temp = cho_order_sku_C.groupby(['user_id', 'o_id'])['o_area'].count().reset_index()
    order_sku_C_count = order_sku_C_count_temp.groupby(['user_id'])['o_id'].count().reset_index()
    order_sku_C_count.columns = ['user_id', 'cate_order_count_%d' % a_day]
    order_sku_C_num = cho_order_sku_C.groupby(['user_id'])['o_sku_num'].sum().reset_index()
    order_sku_C_num.columns = ['user_id', 'cate_order_num_%d' % a_day]

    order_sku_feature = user_price.merge(user_para, on='user_id', how='left'). \
        merge(user_money, on='user_id', how='left'). \
        merge(sku_C_near_num, on='user_id', how='left'). \
        merge(sku_C_near_money, on='user_id', how='left'). \
        merge(order_sku_C_near, on='user_id', how='left'). \
        merge(order_sku_C_count, on='user_id', how='left'). \
        merge(order_sku_C_num, on='user_id', how='left'). \
        merge(order_sku_C_day, on='user_id', how='left')
    return order_sku_feature


def past_action(start_date, end_date, a_day):
    print('processing action data....')
    action_day_diff1 = user_action_table.loc[user_action_table['a_date'] < end_date]
    action_day_diff2 = action_day_diff1.loc[action_day_diff1['a_date'] >= start_date]

    action_day_diff2['past_day'] = action_day_diff2['a_date'].apply(lambda x: (end_date - x).days)
    action_day_diff2 = action_day_diff2.sort_values(by=['user_id', 'past_day'])
    user_action_near_temp = action_day_diff2.groupby('user_id').head(1)
    user_action_near = user_action_near_temp[['user_id', 'past_day']].copy()
    user_action_near.columns = ['user_id', 'action_near_day_%d' % a_day]
    action_num = action_day_diff2.groupby(['user_id'])['a_num'].sum().reset_index()
    action_num.columns = ['user_id', 'action_num_%d' % a_day]
    action_day_temp = action_day_diff2.groupby(['user_id', 'a_date'])['sku_id'].count().reset_index()
    action_day = action_day_temp.groupby(['user_id'])['a_date'].count().reset_index()
    action_day.columns = ['user_id', 'action_day_%d' % a_day]
    action_sku_temp = action_day_diff2.groupby(['user_id', 'sku_id'])['a_date'].count().reset_index()
    action_sku = action_sku_temp.groupby(['user_id'])['sku_id'].count().reset_index()
    action_sku.columns = ['user_id', 'action_sku_count_%d' % a_day]
    action_type_temp = action_day_diff2.groupby(['user_id', 'a_type'])['a_num'].sum().reset_index()
    action_type = action_type_temp.pivot(index='user_id', columns='a_type', values='a_num').reset_index()
    action_type.columns = ['user_id', 'type1_num_%d' % a_day, 'type2_num_%d' % a_day]
    action_type = action_type.fillna(0)


    action_feature = user_action_near.merge(action_num, on='user_id', how='left'). \
        merge(action_day, on='user_id', how='left'). \
        merge(action_sku, on='user_id', how='left'). \
        merge(action_type, on='user_id', how='left')
    return action_feature


def past_action_sku(start_date, end_date, a_day):
    print('processing action_sku data...')
    action_sku_day_diff1 = action_sku_table.loc[action_sku_table['a_date'] < end_date]
    action_sku_day_diff2 = action_sku_day_diff1.loc[action_sku_day_diff1['a_date'] >= start_date]
    action_sku_day_diff2['past_day'] = action_sku_day_diff2['a_date'].apply(lambda x: (end_date - x).days)

    user_price = action_sku_day_diff2.groupby(['user_id'])['price'].agg(['mean', 'max', 'min']).reset_index()
    user_price.columns = ['user_id', 'action_price_mean_%d' % a_day, 'action_price_max_%d' % a_day,
                          'action_price_min_%d' % a_day]
    user_para = action_sku_day_diff2.groupby(['user_id'])['para_1'].agg(['mean', 'max', 'min']).reset_index()
    user_para.columns = ['user_id', 'action_para1_mean_%d' % a_day, 'action_para1_max_%d' % a_day,
                         'action_para1_min_%d' % a_day]

#    cho_action_sku_C1 = action_sku_day_diff2.loc[action_sku_day_diff2['cate'] == 101]
#    cho_action_sku_C2 = action_sku_day_diff2.loc[action_sku_day_diff2['cate'] == 30]
#    cho_action_sku_C = pd.concat([cho_action_sku_C1, cho_action_sku_C2])
    cho_action_sku_C1 = action_sku_day_diff2.loc[action_sku_day_diff2['cate'] == 1]
    cho_action_sku_C2 = action_sku_day_diff2.loc[action_sku_day_diff2['cate'] == 46]
    cho_action_sku_C3 = action_sku_day_diff2.loc[action_sku_day_diff2['cate'] == 71]
    cho_action_sku_C4 = action_sku_day_diff2.loc[action_sku_day_diff2['cate'] == 83]
    cho_action_sku_C = pd.concat([cho_action_sku_C1, cho_action_sku_C2,cho_action_sku_C3,cho_action_sku_C4])
    cho_action_sku_C = cho_action_sku_C.sort_values(by=['user_id', 'past_day'])
    action_sku_C_near_temp = cho_action_sku_C.groupby(['user_id']).head(1)
    action_sku_C_near = action_sku_C_near_temp[['user_id', 'past_day']].copy()
    action_sku_C_near.columns = ['user_id', 'action_cate_past_day_%d' % a_day]
    action_sku_C_num = cho_action_sku_C.groupby(['user_id'])['a_num'].sum().reset_index()
    action_sku_C_num.columns = ['user_id', 'cate_action_num_%d' % a_day]
    action_sku_C_day_temp = cho_action_sku_C.groupby(['user_id', 'a_date'])['sku_id'].count().reset_index()
    action_sku_C_day = action_sku_C_day_temp.groupby(['user_id'])['a_date'].count().reset_index()
    action_sku_C_day.columns = ['user_id', 'action_sku_day_%d' % a_day]


    action_sku_C_type_temp = cho_action_sku_C.groupby(['user_id', 'a_type'])['a_num'].sum().reset_index()
    action_sku_C_type = action_sku_C_type_temp.pivot(index='user_id', columns='a_type', values='a_num').reset_index()
    action_sku_C_type.columns = ['user_id', 'cate_type1_num_%d' % a_day, 'cate_type2_num_%d' % a_day]
    action_sku_C_type = action_sku_C_type.fillna(0)

    action_sku_feature = user_price.merge(user_para, on='user_id', how='left'). \
        merge(action_sku_C_near, on='user_id', how='left'). \
        merge(action_sku_C_num, on='user_id', how='left'). \
        merge(action_sku_C_day, on='user_id', how='left'). \
        merge(action_sku_C_type, on='user_id', how='left')
    return action_sku_feature


def past_comment(start_date, end_date, a_day):
    print('processing comment data...')
    comment_day_diff1 = comment_table.loc[comment_table['comment_create_tm'] < end_date]
    comment_day_diff2 = comment_day_diff1.loc[comment_day_diff1['comment_create_tm'] >= start_date]

    comment_day_diff2['past_day'] = comment_day_diff2['comment_create_tm'].apply(lambda x: (end_date - x).days)
    comment_day_diff2 = comment_day_diff2.sort_values(by=['user_id', 'past_day'])
    comment_near_temp = comment_day_diff2.groupby('user_id').head(1)
    comment_near = comment_near_temp[['user_id', 'past_day']].copy()
    comment_near.columns = ['user_id', 'comment_near_day_%d' % a_day]

    comment_count_temp = comment_day_diff2.groupby(['user_id', 'o_id'])['score_level'].count().reset_index()
    comment_count = comment_count_temp.groupby(['user_id'])['o_id'].count().reset_index()
    comment_count.columns = ['user_id', 'comment_count_%d' % a_day]
    comment_day_temp = comment_day_diff2.groupby(['user_id', 'comment_create_tm'])['score_level'].count().reset_index()
    comment_day = comment_day_temp.groupby(['user_id'])['comment_create_tm'].count().reset_index()
    comment_day.columns = ['user_id', 'comment_day_%d' % a_day]
    comment_score_mean = comment_day_diff2.groupby(['user_id'])['score_level'].mean().reset_index()
    comment_score_mean.columns = ['user_id', 'score_level_mean_%d' % a_day]



    comment_feature = comment_near.merge(comment_count, on='user_id', how='left'). \
        merge(comment_day, on='user_id', how='left'). \
        merge(comment_score_mean, on='user_id', how='left')
    return comment_feature


def past_order_comment(start_date, end_date, a_day):
    print('processing order_comment data...')
    comment_day_diff1 = order_comment_level.loc[order_comment['o_date'] < end_date]
    comment_day_diff2 = comment_day_diff1.loc[comment_day_diff1['o_date'] >= start_date]
    order_comment_score_mean = comment_day_diff2.groupby(['user_id'])['score_level'].mean().reset_index()
    order_comment_score_mean.columns = ['user_id', 'order_score_level_mean_%d' % a_day]
    return order_comment_score_mean


def order_sku_comment(start_date, end_date, a_day):
    print('processing order_sku_comment data...')
    sku_comment_day_diff1 = sku_comment.loc[sku_comment['o_date'] < end_date]
    sku_comment_day_diff2 = sku_comment_day_diff1.loc[sku_comment_day_diff1['o_date'] >= start_date]
    sku_comment_day_diff3 = sku_comment_day_diff2.loc[sku_comment_day_diff2['cate'] == 101]
    sku_comment_day_diff4 = sku_comment_day_diff2.loc[sku_comment_day_diff2['cate'] == 30]
    sku_comment_day_diff = pd.concat([sku_comment_day_diff3, sku_comment_day_diff4])
    sku_comment_score_mean = sku_comment_day_diff.groupby(['user_id'])['score_level'].mean().reset_index()
    sku_comment_score_mean.columns = ['user_id', 'sku_score_level_mean_%d' % a_day]
    return sku_comment_score_mean


def past_order_action(start_date, end_date, a_day):
    order_day_diff1 = user_order_table.loc[user_order_table['o_date'] < end_date]
    order_day_diff2 = order_day_diff1.loc[order_day_diff1['o_date'] >= start_date]
    order_day_diff3 = order_day_diff2[['user_id', 'sku_id', 'o_id', 'o_date']].drop_duplicates()

    order_action_table = order_day_diff3.merge(user_action, on=['user_id', 'sku_id'], how='left')
    order_action_table = order_action_table.loc[order_action_table['a_date'] < order_action_table['o_date']]
    order_action_table['day_diff'] = order_action_table['o_date'] - order_action_table['a_date']
    order_action_table['day_diff'] = order_action_table['day_diff'].apply(lambda x: x.days)
    cho_order_action = order_action_table.loc[order_action_table['day_diff'] <= a_day]

    one_sku_temp1 = cho_order_action.groupby(['user_id', 'sku_id', 'o_id', 'o_date', 'a_type'])[
        'a_num'].sum().reset_index()
    one_sku_type1 = one_sku_temp1.loc[one_sku_temp1['a_type'] == 1]
    one_sku_type1 = one_sku_type1.drop(['a_type'], axis=1)
    one_sku_type2 = one_sku_temp1.loc[one_sku_temp1['a_type'] == 2]
    one_sku_type2 = one_sku_type2.drop(['a_type'], axis=1)
    one_sku_type = one_sku_type1.merge(one_sku_type2, on=['user_id', 'sku_id', 'o_id', 'o_date'], how='outer')
    one_sku_type = one_sku_type.fillna(0)
    one_sku_cate = one_sku_type.merge(sku_basic[['sku_id', 'cate']], on='sku_id', how='left')
    one_sku_cate1 = one_sku_cate.loc[one_sku_cate['cate'] == 101]
    one_sku_cate2 = one_sku_cate.loc[one_sku_cate['cate'] == 30]
    cho_sku_cate = pd.concat([one_sku_cate1, one_sku_cate2])

    one_sku = one_sku_type.groupby(['user_id'])['a_num_x', 'a_num_y'].mean().reset_index()
    one_sku.columns = ['user_id', 'ratio_type1_num_%d' % a_day, 'ratio_type2_num_%d' % a_day]
    one_o_id_temp = one_sku_type.groupby(['user_id', 'o_id'])['a_num_x', 'a_num_y'].sum().reset_index()
    one_o_id = one_o_id_temp.groupby(['user_id'])['a_num_x', 'a_num_y'].mean().reset_index()
    one_o_id.columns = ['user_id', 'order_ratio_type1_num_%d' % a_day, 'order_ratio_type2_num_%d' % a_day]

    cho_one_sku = cho_sku_cate.groupby(['user_id'])['a_num_x', 'a_num_y'].mean().reset_index()
    cho_one_sku.columns = ['user_id', 'cho_ratio_type1_num_%d' % a_day, 'cho_ratio_type2_num_%d' % a_day]
    cho_one_o_id_temp = cho_sku_cate.groupby(['user_id', 'o_id'])['a_num_x', 'a_num_y'].sum().reset_index()
    cho_one_o_id = cho_one_o_id_temp.groupby(['user_id'])['a_num_x', 'a_num_y'].mean().reset_index()
    cho_one_o_id.columns = ['user_id', 'cho_order_ratio_type1_num_%d' % a_day,
                            'cho_order_ratio_type2_num_%d' % a_day]

    order_action_feature = one_sku.merge(one_o_id, on='user_id', how='left'). \
        merge(cho_one_sku, on='user_id', how='left'). \
        merge(cho_one_o_id, on='user_id', how='left')
    return order_action_feature


def get_data_part(end_date, a_day, group):
    print('getting part data  before {}...' .format(end_date))
    end_date = pd.to_datetime(end_date)
    start_date = end_date - timedelta(days=a_day)
#    order_label_part = order_label.loc[order_label['o_month'] == month]
    order_label_part = order_label.loc[order_label['o_date'] < end_date][order_label['o_date'] > end_date - timedelta(days=31)]
    order_label_part = user_basic.merge(order_label_part,how = 'left',on = 'user_id')

    # order_feature = past_order_process(start_date, end_date, a_day)
    # order_sku_feature = past_order_sku(start_date, end_date, a_day)
    start_date1 = end_date - timedelta(days=45)
    start_date2 = end_date - timedelta(days=78)
    start_date3 = end_date - timedelta(days=90)
    start_date4 = end_date - timedelta(days=121)
    start_date5 = end_date - timedelta(days=153)
    start_date6 = end_date - timedelta(days=179)
    start_date7 = end_date - timedelta(days=210)
    start_date8 = end_date - timedelta(days=255)
    start_date9 = end_date - timedelta(days=300)
    order_feature1 = past_order_process(start_date1, end_date, 45)
    order_sku_feature1 = past_order_sku(start_date1, end_date, 45)
    order_feature2 = past_order_process(start_date2, end_date, 78)
    order_sku_feature2 = past_order_sku(start_date2, end_date, 78)
    order_feature3 = past_order_process(start_date3, end_date, 90)
    order_sku_feature3 = past_order_sku(start_date3, end_date, 90)
    order_feature4 = past_order_process(start_date4, end_date, 121)
    order_sku_feature4 = past_order_sku(start_date4, end_date, 121)
    order_feature5 = past_order_process(start_date5, end_date, 153)
    order_sku_feature5 = past_order_sku(start_date5, end_date, 153)
    order_feature6 = past_order_process(start_date6, end_date, 179)
    order_sku_feature6 = past_order_sku(start_date6, end_date, 179)
    order_feature7 = past_order_process(start_date7, end_date, 210)
    order_sku_feature7 = past_order_sku(start_date7, end_date, 210)
    order_feature8 = past_order_process(start_date8, end_date, 255)
    order_sku_feature8 = past_order_sku(start_date8, end_date, 255)
    order_feature9 = past_order_process(start_date9, end_date, 300)
    order_sku_feature9 = past_order_sku(start_date9, end_date, 300)
    order_feature = order_feature1.merge(order_feature2, on='user_id', how='left'). \
        merge(order_feature3, on='user_id', how='left'). \
        merge(order_feature4, on='user_id', how='left'). \
        merge(order_feature5, on='user_id', how='left'). \
        merge(order_feature6, on='user_id', how='left'). \
        merge(order_feature7, on='user_id', how='left'). \
        merge(order_feature8, on='user_id', how='left'). \
        merge(order_feature9, on='user_id', how='left')

    order_sku_feature = order_sku_feature1.merge(order_sku_feature2, on='user_id', how='left'). \
        merge(order_sku_feature3, on='user_id', how='left'). \
        merge(order_sku_feature4, on='user_id', how='left'). \
        merge(order_sku_feature5, on='user_id', how='left'). \
        merge(order_sku_feature6, on='user_id', how='left'). \
        merge(order_sku_feature7, on='user_id', how='left'). \
        merge(order_sku_feature8, on='user_id', how='left'). \
        merge(order_sku_feature9, on='user_id', how='left')

    action_feature = past_action(start_date, end_date, a_day)
    action_sku_feature = past_action_sku(start_date, end_date, a_day)
    comment_feature = past_comment(start_date, end_date, a_day)
    order_comment_score_mean = past_order_comment(start_date, end_date, a_day)
    sku_comment_score_mean = order_sku_comment(start_date, end_date, a_day)
    order_action_feature = past_order_action(start_date, end_date, a_day)

    data_part = order_label_part.merge(order_feature, on='user_id', how='left'). \
        merge(order_sku_feature, on='user_id', how='left'). \
        merge(action_feature, on='user_id', how='left'). \
        merge(action_sku_feature, on='user_id', how='left'). \
        merge(comment_feature, on='user_id', how='left'). \
        merge(order_comment_score_mean, on='user_id', how='left'). \
        merge(sku_comment_score_mean, on='user_id', how='left'). \
        merge(order_action_feature, on='user_id', how='left')

    fill_cols = ['ratio_type1_num_90', 'ratio_type2_num_90', 'order_ratio_type1_num_90', \
                 'order_ratio_type2_num_90', 'cho_ratio_type1_num_90', 'cho_ratio_type2_num_90', \
                 'cho_order_ratio_type1_num_90', 'cho_order_ratio_type2_num_90', 'comment_count_90', \
                 'comment_day_90', 'cate_action_num_90', 'action_sku_day_90', 'cate_type1_num_90', \
                 'cate_type2_num_90', 'action_near_day_90', 'action_num_90', 'action_day_90', \
                 'action_sku_count_90', 'type1_num_90', 'type2_num_90', 'cate_sku_near_num_90', \
                 'cate_past_day_90', 'cate_order_count_90', 'cate_order_num_90', 'order_near_day_90', \
                 'order_count_90', 'order_num_90', 'order_area_num_90', 'order_day_90', 'order_sku_count_90',\
                 'cate_sku_near_num_45', 'cate_past_day_45', 'cate_order_count_45', 'cate_order_num_45', \
                 'order_near_day_45', 'order_count_45', 'order_num_45', 'order_area_num_45', 'order_day_45', \
                 'order_sku_count_45', 'cate_sku_near_num_78', 'cate_past_day_78', 'cate_order_count_78', \
                 'cate_order_num_78', 'order_near_day_78', 'order_count_78', 'order_num_78', 'order_area_num_78', \
                 'order_day_78', 'order_sku_count_78', 'cate_sku_near_num_121', 'cate_past_day_121', \
                 'cate_order_count_121', 'cate_order_num_121', 'order_near_day_121', 'order_count_121', \
                 'order_num_121', 'order_area_num_121', 'order_day_121', 'order_sku_count_121', \
                 'cate_sku_near_num_153', 'cate_past_day_153', 'cate_order_count_153', 'cate_order_num_153', \
                 'order_near_day_153', 'order_count_153', 'order_num_153', 'order_area_num_153', 'order_day_153', \
                 'order_sku_count_153', 'cate_sku_near_num_179', 'cate_past_day_179', 'cate_order_count_179', 'cate_order_num_179', \
                 'order_near_day_179', 'order_count_179', 'order_num_179', 'order_area_num_179', 'order_day_179',\
                 'order_sku_count_179', 'cate_sku_near_num_210', 'cate_past_day_210', 'cate_order_count_210',\
                 'cate_order_num_210', 'order_near_day_210', 'order_count_210', 'order_num_210', 'order_area_num_210',\
                 'order_day_210', 'order_sku_count_210', 'cate_sku_near_num_255', 'cate_past_day_255',\
                 'cate_order_count_255', 'cate_order_num_255', 'order_near_day_255', 'order_count_255',\
                 'order_num_255', 'order_area_num_255', 'order_day_255', 'order_sku_count_255', 'cate_sku_near_num_300',\
                 'cate_past_day_300', 'cate_order_count_300', 'cate_order_num_300', 'order_near_day_300', 'order_count_300',\
                 'order_num_300', 'order_area_num_300', 'order_day_300', 'order_sku_count_300']
    for fill_col in fill_cols:
        data_part[fill_col] = data_part[fill_col].fillna(0)
    data_part = data_part.fillna(-999)
    data_part['CreateGroup'] = group
    return data_part


def get_month_data(a_day):
#    data_part6 = get_data_part(6, '2016-06-01', a_day)
#    data_part7 = get_data_part(7, '2016-07-01', a_day)
#    data_part8 = get_data_part(8, '2016-08-01', a_day)
#    data_part9 = get_data_part(9, '2016-09-01', a_day)
#    data_part10 = get_data_part(10, '2016-10-01', a_day)
#    data_part11 = get_data_part(11, '2016-11-01', a_day)
#    data_part12 = get_data_part(12, '2016-12-01', a_day)
    data_part2 = get_data_part('2017-02-05', a_day, 156)
    data_part3 = get_data_part('2017-03-05', a_day, 186)
    data_part4 = get_data_part('2017-04-04', a_day, 216)
    data_part5 = get_data_part('2017-05-04', a_day, 246)
    data_part6 = get_data_part('2017-06-03', a_day, 276)
    data_part7 = get_data_part('2017-07-03', a_day, 306)
    data_part8 = get_data_part('2017-08-02', a_day, 336)
    data_part9 = get_data_part('2017-09-01', a_day, 366)
    month_data = pd.concat([data_part9, data_part8, data_part7, data_part6, data_part5, data_part4, data_part3, data_part2], axis=0)
    return month_data


def get_user_basic():
    age = pd.get_dummies(user_basic['age'], prefix='age')
    sex = pd.get_dummies(user_basic['sex'], prefix='sex')
    lv_cd = pd.get_dummies(user_basic['user_lv_cd'], prefix='lv_cd')
    user_feature = pd.concat([user_basic, age, sex, lv_cd], axis=1)
    return user_feature


if __name__ == '__main__':
    data_feature = get_month_data(90)
#    user_info = get_user_basic()
#    train_data = data_feature.merge(user_info, on='user_id', how='left')
#    train_data.to_csv('D.csv', index=False)
    data_feature.to_csv('D2b_other.csv', index=False)
