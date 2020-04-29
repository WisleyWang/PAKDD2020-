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
def research_nan():
    paths = ['disk_sample_smart_log_201707.csv', 'disk_sample_smart_log_201708.csv', 'disk_sample_smart_log_201709.csv'
        , 'disk_sample_smart_log_201710.csv', 'disk_sample_smart_log_201711.csv', 'disk_sample_smart_log_201712.csv'
        , 'disk_sample_smart_log_201801.csv', 'disk_sample_smart_log_201802.csv', 'disk_sample_smart_log_201803.csv',
             'disk_sample_smart_log_201804.csv', 'disk_sample_smart_log_201805.csv', 'disk_sample_smart_log_201806.csv',
             'disk_sample_smart_log_201807.csv']
    datas = pd.read_csv('disk_sample_smart_log_201707.csv', index_col=0, chunksize=100)
    for data in datas:
        nafeature = [k for k in data.columns if k not in ['serial_number', 'manufacturer', 'model', 'dt']]
        final_na = nafeature.copy()
        break
    for p in paths:
        path='../data/round1_train/'+p
        print(path)
        datas = pd.read_csv(path, index_col=0, chunksize=50000)
        for data in datas:
            for col in nafeature:
                if col in nafeature:
                    uniq = data[col].nunique()
                    if uniq > 1 and col in final_na:
                        final_na.remove(col)
        # print(path+':{}'.format(nafeature))
        del datas
        gc.collect()
        return final_na
### ------ 减少内存
def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
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

### ----生成serial文件，记录所有硬盘最早的log
def generate_serial():
    datas = ['disk_sample_smart_log_201707.csv', 'disk_sample_smart_log_201708.csv', 'disk_sample_smart_log_201709.csv'
        , 'disk_sample_smart_log_201710.csv', 'disk_sample_smart_log_201711.csv', 'disk_sample_smart_log_201712.csv'
        , 'disk_sample_smart_log_201801.csv', 'disk_sample_smart_log_201802.csv', 'disk_sample_smart_log_201803.csv',
             'disk_sample_smart_log_201804.csv', 'disk_sample_smart_log_201805.csv', 'disk_sample_smart_log_201806.csv',
             'disk_sample_smart_log_201807.csv']
    tests=['../data/round1_testA/disk_sample_smart_log_test_a.csv', '../data/round1_testB/disk_sample_smart_log_test_b.csv']
    datas=['../data/round1_train/'+f for f in datas ]
    paths=datas+tests
    # ['smart_193raw','smart_7raw'，'smart_4raw'] smart_197raw
    serial = pd.DataFrame()
    init_col_name = ['smart_192raw','smart_193raw','smart_195_normalized','smart_241raw','smart_7raw','dt']
    # ['dt','smart_1raw','smart_194raw','smart_195_normalized','smart_241raw']# 'smart_1raw','smart_194raw','smart_195_normalized','init_smart_241raw'
    for path in paths:
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

## 处理data的记录时间戳，获得天，年，月，并将ts转换为时间格式
def procese_dt(data):
    data=reduce_mem(data)
    data.dt=data.dt.apply(lambda x :datetime.datetime(x//10000,(x//100)%100,x%100).strftime('%Y-%m-%d'))
    data['dt']=pd.to_datetime(data.dt)
    data=data.sort_values('dt').reset_index(drop=True)
    return data
# 这里的df 需要先和tag 做merger 得到fault_time

def get_label(df,rate=3,mult=5,sample=False):
    # df.fault_time=df.fault_time.fillna('2019-01-01') # 给未坏的硬盘一个超前的日期
    df['gap_bad_day']=(df['fault_time']-df['dt']).dt.days
    df['label'] =0
    df.loc[df['gap_bad_day']<=30,'label']=1
    # 这里看是设置gap少于多少天为正样本好
    if sample : # 筛选模式
        positive_data=df.loc[df['gap_bad_day']<=60]  # 这里做了点小改动，就是保证一定的边界样本
        negative_data=df.loc[df['label']==0]
        df=pd.DataFrame()
        for k in range(mult):
            f_tmp=negative_data.loc[random.sample(negative_data.index.tolist(),positive_data.shape[0]*rate)]
            df=pd.concat([df,positive_data,f_tmp],axis=0)
        df.reset_index(drop=True)
    # df.loc[(df['gap_bad_day']>30)|pd.isna(df['gap_bad_day']),'label']=30
    # df.loc[:,'label']=1-df['label']/30
    del df['gap_bad_day'],df['fault_time']
    #  df['gap_bad_day']>=0) &
    # overdu_index=df[df['gap_bad_day']<0].index.tolist()
    # print('delect data number:',len(overdu_index))
    # df.drop(overdu_index,inplace=True) # 删除间隔为负数的数据
    # df.reset_index(drop=True,inplace=True)  # 重新索引
    return df

## 计算损失率
def count_nan(df):
    tmp=df[['serial_number','model']]
    tmp['count']=1
    df['log_cusum']=tmp.groupby(['serial_number','model'])['count'].transform(np.cumsum)
    df['miss_data_rate']=1-df['log_cusum']/((df['dt']-df['init_dt']).dt.days+0.0001)
    return df
### 计算所有从差分特征，删除原始特征
def diff_test(df):
    # 根据特征重要性来
    # 输入的数据必须是已经按照时间排好序的 , smart_190_normalized，'smart_195_normalized','smart_192raw'
    fts=['smart_188raw','smart_193raw','smart_192raw','smart_195_normalized','smart_187raw','smart_7_normalized']
    tmp=df.loc[:,['serial_number','model','dt']+fts].groupby(['serial_number','model'])
    average=pd.DataFrame()
    for i in range(1,8,3):
        diffs=['diff_{}_'.format(i)+col for col in fts]
        average[diffs]=tmp[fts].shift(0)-tmp[fts].shift(i)
        # tmp2=sort_df.loc[:,['serial_number','model','dt']+diffs].drop_duplicates(['serial_number','model','dt']).reset_index(drop=True)
        # df=df.merge(tmp2, on=['serial_number','model','dt'], how='left')
    del tmp
    for k in fts:
        df['mean_diff_'+k]=average[['diff_1_'+k,'diff_4_'+k,'diff_7_'+k]].mean(axis=1)
    del average
    return df
### 计算所有特征的初始差值：和serial配合使用
def init_test(df):
    df['server_time'] = (df['dt'] - df['init_dt']).dt.days
    diff_init=['smart_192raw','smart_193raw','smart_195_normalized','smart_241raw','smart_7raw']
    for c in diff_init:
        df['init_diff_'+c]=df[c]-df['init_'+c]
        df.drop('init_'+c,axis=1,inplace=True)
        df.drop(c,axis=1,inplace=True)
    gc.collect()
    return df

## 这些是插值的函数--------------------------------------------------
### 填充nan
def cube_fill(df,model=1):
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
    col_feature=[k for k in df.columns if k not in ['serial_number', 'manufacturer', 'model','dt']]
    for i in tqdm(col_feature):
        # print(i)
        tmp=df.groupby(['serial_number','model'])[i]
        df[i]=tmp.transform(chazhi)
    return df
def chazhi(x):
    org_y=np.array(x)
    new_x=np.array(range(len(x)))
    old_x=new_x[pd.notna(x)]
    old_y=org_y[pd.notna(x)]
    try:
        f=interpolate.interp1d(old_x,old_y,kind='cubic')
        return f(new_x)
    except:
        return x
def generate_data():
    ## 集合3 4 5 6 7 8 月的数据，处理时间，同时补充nan值
    datas = ['disk_sample_smart_log_201803.csv','disk_sample_smart_log_201804.csv',
             'disk_sample_smart_log_201805.csv', 'disk_sample_smart_log_201806.csv',
             'disk_sample_smart_log_201807.csv']
    tests = ['../data/round1_testA/disk_sample_smart_log_test_a.csv',
             '../data/round1_testB/disk_sample_smart_log_test_b.csv']
    datas = ['../data/round1_train/' + f for f in datas]
    paths = datas + tests
    ## 已筛选特征，这里是直接列出来了
    efficien_col=['serial_number', 'manufacturer', 'model', 'smart_1_normalized',
       'smart_1raw', 'smart_3_normalized', 'smart_4raw', 'smart_5raw',
       'smart_7_normalized', 'smart_7raw', 'smart_9_normalized', 'smart_9raw',
       'smart_12raw', 'smart_184raw', 'smart_187_normalized', 'smart_187raw',
       'smart_188raw', 'smart_189_normalized', 'smart_189raw',
       'smart_190_normalized', 'smart_191_normalized', 'smart_191raw',
       'smart_192raw', 'smart_193_normalized', 'smart_193raw', 'smart_194raw',
       'smart_195_normalized', 'smart_195raw', 'smart_197raw', 'smart_198raw',
       'smart_199raw', 'smart_240raw', 'smart_241raw', 'smart_242raw', 'dt']
    data_345678 = pd.DataFrame()
    for path in paths:
        data = pd.read_csv(path)  # 读取
        data = data[efficien_col]
        data = procese_dt(data)
        data_345678 = data_345678.append(data).reset_index(drop=True)

    data_345678.sort_values('dt').reset_index(drop=True)
    print('cube for data....')
    data_345678 = cube_fill(data_345678)  # 差值raw
    print('save data to  /user_data/cube_data')
    data_345678.to_pickle('../user_data/cube_data/data_345678.pkl')
    del data_345678
    gc.collect()
import os
if __name__ == '__main__':
    #生成插值后的data345678
    pd.DataFrame().dropna()
    print('generate data ......')
    generate_data()
    print('generate serial ....')
    generate_serial()
    ## 读取data------------------
    print('读取data------------------')
    data_345678 = pd.read_pickle('../user_data/cube_data/data_345678.pkl')
    ## 读取tag-----------------------
    print('读取tag-----------------------')
    tag = pd.read_csv('../data/round1_train/fault_tag.csv')
    tag['fault_time'] = pd.to_datetime(tag['fault_time'])
    ###tag表里面有的硬盘同一天发生几种故障
    tag['tag'] = tag['tag'].astype(str)
    # 去掉4 5 6 标签的样本
    part_tag = tag[(tag['tag'] != '4') & (tag['tag'] != '6') & (tag['tag'] != '5')]  # 这份标记用于训练集
    tag = tag.groupby(['serial_number', 'fault_time', 'model'])['tag'].apply(lambda x: '|'.join(x)).reset_index()
    ## 读取serial---------------
    print('读取serial---------------')
    serial = pd.read_csv('../user_data/tmp_data/serial.csv')
    serial.init_dt = pd.to_datetime(serial.init_dt)

    #计算特征：
    print('calculate diff----')
    data_345678 = diff_test(data_345678)
    ## 计算 Init---
    print('calculate Init----')
    data_345678 = data_345678.merge(serial, how='left', on=['serial_number', 'model'])  #
    data_345678 = init_test(data_345678)
    ## 计算 数据缺损率
    print('calculate nan rate ----')
    data_345678 = count_nan(data_345678)
    # ## 打标签同时赛选样本
    print('tag label ----')
    data_345678 = data_345678.merge(tag[['serial_number', 'model', 'fault_time']], how='left',
                                    on=['serial_number', 'model'])  # 合并
    data_345678 = get_label(data_345678, sample=False)  #

    data_345678['serial_number'] = data_345678['serial_number'].apply(lambda x: int(x.split('_')[1]))  # 处理serial
    print('save data for train ....')
    data_345678.to_pickle('../user_data/process_data/process_data_345678.pkl')