import pandas as pd
import datetime
import numpy as np
import gc
from scipy import interpolate
from tqdm import tqdm
import random
import pickle
###-----搜索所有无效特征（nan或者只有当值的特征）---------------
## 返回需要剔除的特征数组
import pandas as pd
import numpy as np
import lightgbm as lgb
import datetime
from tqdm import tqdm
import gc
from lightgbm.sklearn import LGBMClassifier
import pickle
import zipfile
import os
from scipy import interpolate

# 'smart_187raw'坏盘才有记录
# 'smart_5_normalized','smart_4_normalized', 'smart_12_normalized',
# 'smart_184_normalized',
# 'smart_188_normalized',
# 'smart_190raw',
# 'smart_192_normalized',
# 'smart_194_normalized',
# 'smart_197_normalized',
# 'smart_198_normalized'
## 处理data的记录时间戳，获得天，年，月，并将ts转换为时间格式
efficien_col = ['serial_number', 'manufacturer', 'model', 'smart_1_normalized',
                'smart_1raw', 'smart_3_normalized', 'smart_4raw', 'smart_5raw',
                'smart_7_normalized', 'smart_7raw', 'smart_9_normalized', 'smart_9raw',
                'smart_12raw', 'smart_184raw', 'smart_187_normalized',
                'smart_188raw', 'smart_189_normalized', 'smart_189raw',
                'smart_190_normalized', 'smart_191_normalized', 'smart_191raw',
                'smart_192raw', 'smart_193_normalized', 'smart_193raw', 'smart_194raw',
                'smart_195_normalized', 'smart_195raw', 'smart_197raw', 'smart_198raw',
                'smart_199raw', 'smart_240raw', 'smart_241raw', 'smart_242raw', 'dt', ]


# 'smart_187raw'坏盘才有记录
# 'smart_5_normalized','smart_4_normalized', 'smart_12_normalized',
# 'smart_184_normalized',
# 'smart_188_normalized',
# 'smart_190raw',
# 'smart_192_normalized',
# 'smart_194_normalized',
# 'smart_197_normalized',
# 'smart_198_normalized'
## 处理data的记录时间戳，获得天，年，月，并将ts转换为时间格式
def procese_dt(data):
    data = reduce_mem(data)
    data.dt = data.dt.apply(lambda x: datetime.datetime(x // 10000, (x // 100) % 100, x % 100).strftime('%Y-%m-%d'))
    data['dt'] = pd.to_datetime(data.dt)
    data = data.sort_values('dt').reset_index(drop=True)
    return data


### ------ 减少内存
def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if (col_type != object) and (col_type != '<M8[ns]'):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


def get_label(df, rate=3, mult=5, sample=False):
    # df.fault_time=df.fault_time.fillna('2019-01-01') # 给未坏的硬盘一个超前的日期
    #     df['fault_time'].fillna(value=datetime.datetime(2019,1,1),inplace=True)
    df['gap_bad_day'] = (df['fault_time'] - df['dt']).dt.days
    df['label'] = 0
    df.loc[df['gap_bad_day'] < 30, 'label'] = 10
    df.loc[df['gap_bad_day'] < 20, 'label'] = 20
    df.loc[df['gap_bad_day'] < 10, 'label'] = 30
    df.loc[df['gap_bad_day'] < 5, 'label'] = 40
    df.loc[df['gap_bad_day'] < 2, 'label'] = 100
    df.loc[df.label != 0, 'label'] = df.loc[df.label != 0, 'label'] + df.loc[df.label != 0, 'tag']
    #     df['label']=np.log1p(df['label'])

    #     df.loc[df['gap_bad_day']<30,'label']=10
    #     df.loc[df['gap_bad_day']<20,'label']=20
    #     df.loc[df['gap_bad_day']<10,'label']=30
    #     df.loc[df['gap_bad_day']<5,'label']=40
    #     df.loc[df['gap_bad_day']<2,'label']=100
    #  36.440
    # 这里看是设置gap少于多少天为正样本好
    #     del df['fault_time']
    return df


### 计算所有从差分特征，删除原始特征
def diff_test(df):
    # 根据特征重要性来
    # 输入的数据必须是已经按照时间排好序的 , smart_190_normalized，'smart_195_normalized','smart_192raw'
    # 位0 的diff  'smart_184raw' 'smart_193_normalized','smart_195raw'
    fts = ['smart_1_normalized',
           'smart_1raw', 'smart_5raw',
           'smart_7_normalized', 'smart_7raw',
           'smart_184raw', 'smart_187_normalized', 'smart_188raw',
           'smart_191_normalized', 'smart_191raw',
           'smart_193raw', 'smart_194raw',
           'smart_195_normalized', 'smart_195raw', 'smart_198raw',
           'smart_199raw']
    # 'smart_1raw','smart_1_normalized','smart_5raw','smart_187_normalized','smart_188raw','smart_195raw','smart_198raw', 'smart_199raw'

    tmp = df.loc[:, ['serial_number', 'model', 'dt'] + fts].groupby(['serial_number', 'model'])
    average = df.loc[:, ['serial_number', 'model']]

    for i in range(1, 8, 3):
        gap = (tmp['dt'].shift(0) - tmp['dt'].shift(i)).dt.days
        for col in fts:
            average['diff_{}_{}'.format(i, col)] = (tmp[col].shift(0) - tmp[col].shift(i)) / gap
    for col in fts:
        average['cumsum_diff_' + col] = average.groupby(['serial_number', 'model'])['diff_1_{}'.format(col)].cumsum()

    #     second_col=[k for k in average.columns if k not in ['serial_number','model']]
    #     for c in second_col:
    #         average['second_'+c]=tmp[c].shift(0)-tmp[c].shift(1)
    #         tmp2=sort_df.loc[:,['serial_number','model','dt']+diffs].drop_duplicates(['serial_number','model','dt']).reset_index(drop=True)
    #     df=df.merge(average, on=['serial_number','model','dt'], how='left')
    del tmp
    for k in fts:
        df['mean_diff_' + k] = average[['diff_{}_'.format(n) + k for n in range(1, 8, 3)]].mean(axis=1)
        df['cumsum_' + k] = average['cumsum_diff_' + k]
    #     average.drop(['serial_number','model'],axis=1,inplace=True)
    #     df=pd.concat([df,average],axis=1)
    del average
    return df


### 计算所有特征的初始差值：和serial配合使用
def init_test(df):
    df['server_time'] = (df['dt'] - df['init_dt']).dt.days
    diff_init = ['smart_1raw', 'smart_3_normalized', 'smart_5raw', 'smart_7_normalized', 'smart_7raw', 'smart_9raw',
                 'smart_12raw',
                 'smart_190_normalized', 'smart_191raw', 'smart_193raw', 'smart_194raw']
    #     #['smart_192raw','smart_193raw','smart_195_normalized','smart_241raw','smart_7raw']
    #  if c not in ['smart_184raw', 'smart_187_normalized','smart_188raw','smart_189raw',
    #        'smart_189_normalized','smart_240raw','smart_241raw','smart_242raw','smart_192raw','smart_1_normalized','smart_4raw']:
    ##'smart_193raw','smart_197raw','smart_198raw'
    for c in diff_init:
        df['init_diff_' + c] = df[c] - df['init_' + c]
        df.drop('init_' + c, axis=1, inplace=True)
    #         df.drop(c,axis=1,inplace=True)
    # #
    gc.collect()
    return df


## c窗函数
def window_feature(df, window=7):
    tmp = df.groupby(['serial_number', 'model']).rolling(window)

    col = ['smart_1_normalized', 'smart_7raw', 'smart_193raw', 'smart_195_normalized']
    # std
    tmp_var = tmp[col].std().reset_index(drop=True)
    tmp_var.columns = ['var_' + i for i in col]
    for c in col:
        df['var_diff_' + c] = df['mean_diff_' + c] * tmp_var['var_' + c]
    #     df=pd.concat([df,tmp_mean,tmp_max,tmp_std],axis=1)
    del tmp
    return df


## 计算损失率
def count_nan(df):
    tmp = df[['serial_number', 'model']]
    tmp['count'] = 1
    df['log_cusum'] = tmp.groupby(['serial_number', 'model'])['count'].transform(np.cumsum)
    df['miss_data_rate'] = 1 - df['log_cusum'] / ((df['dt'] - df['init_dt']).dt.days + 0.0001)
    del df['log_cusum']
    return df


## 预测为提交比赛的格式
# tag为标签文件，predict_df为提交文件,mon是评估的窗口月份
def outline_evalue(tag, predict_df, mon=6):
    tag['fault_time'] = pd.to_datetime(tag['fault_time'])
    # 只取最早的一天为准
    predict_df = predict_df.sort_values('dt')
    predict_df = predict_df.drop_duplicates(['serial_number', 'model'])
    #
    predict_df = predict_df.merge(tag, on=['serial_number', 'model'], how='left')
    predict_df['gap_day'] = (predict_df['fault_time'] - predict_df['dt']).dt.days
    ###npp:评估窗口内被预测出未来30天会坏的盘数
    npp = predict_df.shape[0]
    # ntpp:评估窗口内第一次预测故障的日期后30天内确实发生故障的盘数
    ntpp = predict_df.loc[predict_df['gap_day'] < 30].shape[0]
    # npr: 评估窗口内所有的盘故障数
    npr = sum(tag['fault_time'].dt.month == mon)
    # tpr: 评估窗口内故障盘被提前30天发现的数量
    tpr = predict_df.loc[(predict_df['fault_time'].dt.month == mon)].shape[0]
    ## ntpp/npp
    pricision = ntpp / npp
    # ntpr/npr
    recall = tpr / npr
    print('pricision:', pricision)
    print('recall:', recall)
    f1 = 2 * pricision * recall / (pricision + recall)
    print('f1*100 :', f1 * 100)
    return predict_df


### 填充nan
def cube_fill(df, model=1):
    '''
    model=1 插值raw
    model=0 插值 normal
    '''
    # if model :
    #     print('cubic to raw')
    #     col_feature=['smart_{}raw'.format(i) for i in [1,4,5,7,9,12,184,187,188,189,191,192,193,194,195,197,198,199]]
    # else:
    #     print('cubic to normalized')
    #     col_feature=['smart_{}_normalized'.format(i) for i in [1,3,7,9,187,189,190,191,193,195]]
    col_feature = [k for k in df.columns if k not in ['serial_number', 'manufacturer', 'model', 'dt']]
    for i in tqdm(col_feature):
        # print(i)
        tmp = df.groupby(['serial_number', 'model'])[i]
        df[i] = tmp.transform(chazhi)
    return df


def chazhi(x):
    org_y = np.array(x)
    new_x = np.array(range(len(x)))
    old_x = new_x[pd.notna(x)]
    old_y = org_y[pd.notna(x)]
    try:
        f = interpolate.interp1d(old_x, old_y, kind='cubic')
        return f(new_x)
    except:
        return x


def mark_score(df):
    score_col = ['smart_7raw', 'smart_9raw', 'smart_191raw', 'smart_193raw', 'smart_241raw', 'smart_242raw']
    tmp = df[['serial_number', 'model'] + score_col].groupby(['serial_number', 'model'])
    for i in score_col:
        df[i + '_quantile'] = tmp[i].expanding().quantile(0.75).reset_index(drop=True)
    for i in score_col:
        df[i + '_score'] = (df[i] - df['init_' + i]) / (df[i + '_quantile'] - df['init_' + i] + 1) * 100
        df.drop(i + '_quantile', axis=1, inplace=True)
    return df


def spare_feature(df):
    for i in ['smart_7raw_score', 'smart_9raw_score', 'smart_191raw_score',
              'smart_193raw_score', 'smart_241raw_score', 'smart_242raw_score']:
        tmp = pd.cut(df[i], bins=[-np.inf, -100, -60, -40, -20, 0, 20, 40, 60, 100, np.inf], labels=range(10))
        #     tmp=pd.qcut(df[i].values,q=10,labels=range(10),duplicates='drop')
        df[i + '_level'] = tmp.get_values()
    cross_cols = ['smart_7raw_score_level', 'smart_9raw_score_level', 'smart_191raw_score_level',
                  'smart_193raw_score_level', 'smart_241raw_score_level', 'smart_242raw_score_level']
    for i in range(len(cross_cols)):
        for j in range(i + 1, len(cross_cols)):
            df['cross_{}_{}'.format(cross_cols[i], cross_cols[j])] = np.int8(
                df[cross_cols[i]].values * 10 + df[cross_cols[j]].values)
    df.drop(cross_cols, axis=1, inplace=True)
    gc.collect()
    return df


def gather_erro(df):
    erro_cols = ['smart_187_normalized', 'smart_188raw', 'smart_184raw', 'smart_197raw', 'smart_198raw', 'smart_199raw',
                 'smart_241raw']  # 'smart_187_normalized',

    df['erro_mark'] = 0
    for m in erro_cols:
        df['erro_mark'] = df['erro_mark'] + df[m].values.astype('int32')
    return df


def gct_change(df):
    gct_col = ['smart_7raw', 'smart_3_normalized', 'smart_1raw', 'smart_5raw']  # smart_198raw
    tmp = df.groupby(['serial_number', 'model'])
    for col in gct_col:
        df['pct_' + col] = tmp[col].pct_change()
    return df


def get_mad(df):
    tmp = df.groupby(['serial_number', 'model'])
    mad_col = ['smart_1raw', 'smart_7raw', 'smart_3_normalized', 'smart_4raw', 'smart_9raw', 'smart_198raw']
    for col in mad_col:
        df['mad_' + col] = tmp[col].mad().reset_index(drop=True)
    return df


def ewm_calculate(df):
    ewm_col = ['smart_1raw', 'smart_1_normalized', 'smart_5raw', 'smart_187_normalized', 'smart_188raw', 'smart_195raw',
               'smart_198raw', 'smart_197raw', 'smart_199raw']
    tmp = df.groupby(['serial_number', 'model'])
    for col in tqdm(ewm_col):
        df['ewm_' + col + '_mean'] = tmp[col].transform(lambda x: pd.Series.ewm(x, span=7).mean())
        df['ewm_' + col + '_std'] = tmp[col].transform(lambda x: pd.Series.ewm(x, span=7).std())
    return df


def ewm_var_diff(df):
    for c in ['smart_1raw', 'smart_1_normalized', 'smart_5raw', 'smart_187_normalized', 'smart_188raw', 'smart_195raw',
              'smart_198raw', 'smart_199raw']:
        df['ewm_std_diff_' + c] = df['ewm_' + c + '_std'] * df['mean_diff_' + c]
    return df


def curr_rate(df):
    col = ['smart_4raw', 'smart_12raw', 'smart_192raw']
    df.dt = df['dt'].astype(str)
    tmp = df.groupby('dt')[col].quantile(0.99)
    tmp.reset_index(inplace=True)
    new_name = list(map(lambda x: 'curr_' + x, col))
    tmp.rename(columns=dict(zip(col, new_name)), inplace=True)
    df = df.merge(tmp[['dt'] + new_name], on='dt', how='left')
    del tmp
    for c in col:
        df['curr_rate_' + c] = df[c] / (df['curr_' + c] + 0.1)
        df.drop('curr_' + c, axis=1, inplace=True)
    df.dt = pd.to_datetime(df.dt)
    return df


def scale_smart(df):
    nums = [1, 7, 189, 191, 193, 195]
    for n in nums:
        df['scale_' + str(n)] = df['smart_{}raw'.format(n)] / (df['smart_{}_normalized'.format(n)] + 0.1)
    return df


# def data_smooth(x, alpha=20, beta=1):
#     tmp =x.shift(0)-( x.shift(1) * (alpha - beta) / alpha + x.shift(0) * beta / alpha)
#     return tmp

def data_smoother(df, features, alpha=20, beta=1):
    tmp = df.groupby(['serial_number', 'model'])
    for feature in features:
        df[feature + "_smooth"] = tmp[feature].shift(0) - (
                    tmp[feature].shift(1) * (alpha - beta) / alpha + tmp[feature].shift(0) * beta / alpha)
    return df

def genrate_train_data():
    ### 先构造训练集：
    paths=['disk_sample_smart_log_201803.csv','disk_sample_smart_log_201804.csv',
             'disk_sample_smart_log_201805.csv', 'disk_sample_smart_log_201806.csv',
             'disk_sample_smart_log_201807.csv','disk_sample_smart_log_201808.csv']

    datas = ['../data/round1_train/' + f for f in paths]
    data_45678 = pd.DataFrame()
    for path in datas:
        data=pd.read_csv(path)
        data=data[efficien_col]
        data =procese_dt(data)
        data_45678= data_45678.append(data).reset_index(drop=True)
    data_45678=data_45678.sort_values('dt').reset_index(drop=True)
    #三次样条插值
    data_45678 = cube_fill(data_45678, model=1)
    data_45678.to_pickle('../user_data/cube_data/data_345678.pkl')
### ----生成serial文件，记录所有硬盘最早的log
def generate_serial():
    datas = ['disk_sample_smart_log_201707.csv', 'disk_sample_smart_log_201708.csv', 'disk_sample_smart_log_201709.csv'
        , 'disk_sample_smart_log_201710.csv', 'disk_sample_smart_log_201711.csv', 'disk_sample_smart_log_201712.csv'
        , 'disk_sample_smart_log_201801.csv', 'disk_sample_smart_log_201802.csv', 'disk_sample_smart_log_201803.csv',
             'disk_sample_smart_log_201804.csv', 'disk_sample_smart_log_201805.csv', 'disk_sample_smart_log_201806.csv',
             'disk_sample_smart_log_201807.csv','disk_sample_smart_log_201808.csv']
    datas=['../data/round1_train/'+f for f in datas ]
    # ['smart_193raw','smart_7raw'，'smart_4raw'] smart_197raw
    serial = pd.DataFrame()
    init_col_name = ['smart_1raw','smart_3_normalized','smart_5raw', 'smart_7_normalized',
        'smart_7raw', 'smart_9raw','smart_12raw','smart_190_normalized', 'smart_191raw',
        'smart_193raw','smart_194raw','dt']
    # ['dt','smart_1raw','smart_194raw','smart_195_normalized','smart_241raw']# 'smart_1raw','smart_194raw','smart_195_normalized','init_smart_241raw'
    for path in datas:
        print(path)
        data = pd.read_csv(path, chunksize=500000)
        for d in data:
            d = d[['serial_number', 'model'] + init_col_name].sort_values('dt').drop_duplicates(
                ['serial_number', 'model'])
            serial = pd.concat((serial, d), axis=0)
            serial = serial.sort_values('dt').drop_duplicates(['serial_number', 'model']).reset_index(drop=True)

    serial.dt = pd.to_datetime(
        serial.dt.apply(lambda x: datetime.datetime(x // 10000, (x // 100) % 100, x % 100).strftime('%Y-%m-%d')))

    serial.rename(columns=dict(zip(init_col_name, ['init_' + col for col in init_col_name])), inplace=True)

    serial.to_csv('../user_data/tmp_data/serial.csv', index=None)
if __name__ == '__main__':
    #生成插值后的data345678
    print('generate data ......')
    genrate_train_data()
    print('generate serial ....')
    generate_serial()
    ## 读取data------------------
    print('读取data------------------')
    data_345678 = pd.read_pickle('../user_data/cube_data/data_345678.pkl')
    ## 读取tag-----------------------
    print('读取tag-----------------------')
    tag = pd.read_csv('../data/round1_train/fault_tag.csv')
    tag8 = pd.read_csv('../data/round1_train/disk_sample_fault_tag_201808.csv')
    tag = tag.append(tag8)
    del tag8
    tag['fault_time'] = pd.to_datetime(tag['fault_time'])
    ###tag表里面有的硬盘同一天发生几种故障
    tag['tag'] = tag['tag'].astype(str)
    # 去掉4 5 6 标签的样本
    # part_tag = tag[(tag['tag'] != '4') & (tag['tag'] != '6') & (tag['tag'] != '5')]  # 这份标记用于训练集
    # tag.drop_duplicates(['serial_number', 'fault_time', 'model'],inplace=True)
    # tag = tag.groupby(['serial_number', 'fault_time', 'model'])['tag'].apply(lambda x: '|'.join(x)).reset_index()
    tag = tag.groupby(['serial_number', 'fault_time', 'model'])['tag'].apply(lambda x: x.min()).reset_index()
    tag['tag'] = tag['tag'].map(dict(zip(['0', '6', '4', '2', '1', '3', '5'], range(1, 8))))
    ## 读取serial---------------
    serial = pd.read_csv('../user_data/tmp_data/serial.csv')
    serial.init_dt = pd.to_datetime(serial.init_dt)
    gc.collect()

    #计算特征：
    data_45678 = pd.read_pickle('../user_data/cube_data/data_345678.pkl') #读取
    # 计算 Init---
    print('calculate Init----')
    data_45678 = data_45678.merge(serial, how='left', on=['serial_number', 'model'])  #
    data_45678 = init_test(data_45678)
    # # ## 计算 数据缺损率
    print('calculate nan rate ----')
    data_45678 = count_nan(data_45678)
    data_45678 = gather_erro(data_45678)
    # 计算变化率
    print('calculate gct ----')
    data_45678 = gct_change(data_45678)
    # diff
    print('diff---')
    data_45678 = diff_test(data_45678)
    data_45678 = curr_rate(data_45678)
    print('calculate ewm ----')
    data_45678 = ewm_calculate(data_45678)  # 之前保存了 线下不需要在做了
    data_45678 = ewm_var_diff(data_45678)
    # scale:  raw/nor
    print('scale smart ----')
    data_45678 = scale_smart(data_45678)
    # smooth
    features = ['smart_1_normalized', 'smart_5raw', 'smart_187_normalized', 'smart_188raw', 'smart_195raw',
                'smart_198raw', 'smart_197raw', 'smart_199raw']
    data_45678 = data_smoother(data_45678, features)

    #  # ## 打标签同时赛选样本
    print('tag label ----')
    data_45678 = data_45678.merge(tag[['serial_number', 'model', 'fault_time', 'tag']], how='left',
                                  on=['serial_number', 'model'])  # 合并
    data_45678.loc[pd.isna(data_45678.tag), 'tag'] = 0
    data_45678 = get_label(data_45678, sample=False)  #
    data_45678['serial_number'] = data_45678['serial_number'].apply(lambda x: int(x.split('_')[1]))  # 处理serial


    print('save data for train ....')
    data_345678.to_pickle('../user_data/process_data/process_data_345678.pkl')