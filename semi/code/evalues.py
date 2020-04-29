import pandas as pd
import gc
import pickle
from lightgbm.sklearn import LGBMClassifier
import numpy as np

'''
这部分是用于线下评估，和线上赛选样本的代码，我们尝试了多种线上筛选样本的方式
为了满足赛方要求的当日预测。我们只采用单天选取预测的方式。关于阈值赛选和控制整月disk
数量的方式，其实效果是更好的，但是我们认为这有违背比赛的要求，所以线上并未采取这样的手段，
代码仅供参考
'''
### 用于线下f1评估，仿照赛方规则，label是整月的所有数据
## 预测为提交比赛的格式
# tag为标签文件，predict_df为提交文件,mon是评估的窗口月份
def outline_evalue(tag,predict_df,mon=6):
    tag['fault_time'] = pd.to_datetime(tag['fault_time'])
    # 只取最早的一天为准
    predict_df=predict_df.sort_values('dt')
    predict_df=predict_df.drop_duplicates(['serial_number','model'])
    #
    predict_df=predict_df.merge(tag,on=['serial_number','model'],how='left')
    predict_df['gap_day']=(predict_df['fault_time']-predict_df['dt']).dt.days
    ###npp:评估窗口内被预测出未来30天会坏的盘数
    npp=predict_df.shape[0]
    # ntpp:评估窗口内第一次预测故障的日期后30天内确实发生故障的盘数
    ntpp=predict_df.loc[predict_df['gap_day']<=30].shape[0]
    # npr: 评估窗口内所有的盘故障数
    npr=sum(tag['fault_time'].dt.month==mon)
    # tpr: 评估窗口内故障盘被提前30天发现的数量
    tpr=predict_df.loc[(predict_df['fault_time'].dt.month==mon)].shape[0]
    ## ntpp/npp
    pricision=ntpp/npp
    # ntpr/npr
    recall=tpr/npr
    print('pricision:',pricision)
    print('recall:',recall)
    f1=2*pricision*recall/(pricision+recall)
    print('f1*100 :',f1*100)
    return predict_df

if __name__ == '__main__':
    ## 这里的val_pred 是模型预测出来的回归值，这里我们就以之前训练的一个模型为例。
    with open('../model/ensenble_lgb0_58.pkl', 'rb') as f:
        lgb_model=pickle.load(f)
    with open('tag_model.pkl', 'rb') as f:
        tag_model = pickle.load(f)
    #---加载数据
    data_45678 = pd.read_pickle('../user_data/process_data/process_data_345678.pkl')
    smartcol = sorted([col for col in data_45678.columns if col not in ['dt', 'fault_time',
                                                                        'tag', 'label', 'gap_bad_day', 'manufacturer',
                                                                        'init_dt', 'init_smart_192raw', 'smart_4raw',
                                                                        'smart_195raw'
        , 'smart_12raw', 'smart_1_normalized', 'smart_190_normalized', 'times', 'serial_number', 'model', 'pred_tag']])
    # 'mean_diff_smart_1_normalized' ,,'mean_diff_smart_1raw',
    print(smartcol)
    print(len(smartcol))

    # 7
    val_x = data_45678.loc[data_45678['dt'].dt.month == 7]
    val_y = data_45678.loc[data_45678['dt'].dt.month == 7, 'label']

    test_8 = data_45678.loc[(data_45678['dt'].dt.month == 8)]
    # del data_45678
    gc.collect()

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

    print('value')
    val_tag = np.argmax(tag_model.predict_proba(val_x[smartcol]), axis=1)
    val_x['tag_predict'] = val_tag
    # 添加特征
    smartcol += ['tag_predict']
    #预测：
    val_pred=lgb_model.predict(val_x[smartcol])
#---------------------------------------------------------------------
    ####评估版本1.0 限制整月数量的阈值搜索法（线上要求不能使用）
    ##--------评估 val--------
    t0 = 0.07
    v = 0.02
    best_t = t0
    # test_8['predict']=test_pred
    # sub=test_8[['model', 'serial_number', 'dt', 'predict']]
    val_x['predict'] = val_pred
    sub = val_x[['model', 'serial_number', 'dt', 'predict']]
    for step in range(201):
        curr_t = t0 + step * v
        sub['label'] = [1 if x >= curr_t else 0 for x in sub['predict']]
        curr_len = len(sub.loc[sub.label == 1].drop_duplicates(['serial_number', 'model']))
        print('step: {} threshold: {} number:{}'.format(step, curr_t, curr_len))
        if curr_len <= 200:
            best_t = curr_t
            print('step: {}   best threshold: {}'.format(step, best_t))
            break

    sub['label'] = [1 if x >= best_t else 0 for x in sub['predict']]
    sub = sub.loc[sub.label == 1]
    # sub = sub.sort_values('predict', ascending=False)
    # sub = sub.drop_duplicates(['serial_number', 'model'])

    # sub['label'] = sub['predict'].rank()
    # sub['label'] = (sub['label'] >= sub.shape[0] * 0.998).astype(int)

    print(len(sub))
    nout = len(sub)
    sub.serial_number = sub.serial_number.apply(lambda x: 'disk_' + str(x))
    ## 评估f1
    a = outline_evalue(tag, sub, mon=7)
    print(a['tag'].value_counts())
#-----------------------------------------------------
    ##评估版本1.1 投票法： 当天投票+阈值+回溯：
    val_x['predict'] = val_pred
    # tmp_val['predict']=val_pred
    sub = val_x[['model', 'serial_number', 'dt', 'predict', ]]
    sub['id'] = sub['serial_number'] * 10 + sub['model']
    dayss = sorted(sub['dt'].dt.day.unique())
    final_result = pd.DataFrame()
    select_data = {}
    last_curr = pd.DataFrame()
    throsed = 999
    # 一天天的数据预测
    for d in dayss:
        curr_data = sub.loc[sub['dt'].dt.day == d]
        curr_data['rank'] = curr_data['predict'].rank(ascending=False)
        new_curr = curr_data.loc[curr_data['rank'] < 20]
        new_id = new_curr['id'].values
        for s in new_id:
            select_data[s] = select_data.setdefault(s, 0) + 1
            tmp = new_curr.loc[new_curr['id'] == s]
            if select_data[s] >= 2 or tmp['predict'].values[0] >= throsed:
                final_result = final_result.append(tmp)
        # 跟新阈值
        if final_result.shape[0] > 0:
            #         print(throsed)
            throsed = final_result['predict'].min()
        # 回溯前一天中未被召回的样本
        if last_curr.shape[0] > 0:
            final_result = final_result.append(last_curr.loc[last_curr['predict'] >= throsed])
        last_curr = curr_data
    # 去除重复取的记录
    final_result.drop_duplicates(['serial_number', 'model', 'dt'], inplace=True)
    final_result.serial_number = final_result.serial_number.apply(lambda x: 'disk_' + str(x))

    ## 评估1.2 分model投票法
    ###新策略评估阈值：
    # test_8['predict']=test_pred
    val_x['predict'] = val_pred
    sub = val_x[['model', 'serial_number', 'dt', 'predict', 'gap_bad_day']]
    sub['id'] = sub['serial_number'] * 10 + sub['model']
    dayss = sorted(sub['dt'].dt.day.unique())
    final_result = pd.DataFrame()
    select_data = {}
    last_curr1 = pd.DataFrame()
    last_curr2 = pd.DataFrame()
    throsed = 999
    # 一天天的数据预测
    throsed1 = 999
    throsed2 = 999
    model1_nn = sub.loc[sub.model == 1].shape[0]
    model2_nn = sub.loc[sub.model == 2].shape[0]
    for d in dayss:
        curr_data = sub.loc[sub['dt'].dt.day == d]
        model1_disk = curr_data.loc[curr_data.model == 1]
        model2_disk = curr_data.loc[curr_data.model == 2]
        model1_disk['rank'] = model1_disk['predict'].rank(ascending=False)
        model2_disk['rank'] = model2_disk['predict'].rank(ascending=False)

        new_curr = pd.DataFrame()

        nn1 = np.round(340 * model1_disk.shape[0] / model1_nn)
        nn2 = np.round(340 * model2_disk.shape[0] / model2_nn)
        print('nn1:{:.2f} nn2:{:.2f}'.format(nn1, nn2))
        new_curr = new_curr.append(model1_disk.loc[model1_disk['rank'] < nn1])
        new_curr = new_curr.append(model2_disk.loc[model2_disk['rank'] < nn2])
        new_id = new_curr['id'].values
        for s in new_id:
            select_data[s] = select_data.setdefault(s, 0) + 1
            tmp = new_curr.loc[new_curr['id'] == s]
            if select_data[s] >= 2:
                final_result = final_result.append(tmp)
            elif (tmp['model'].values[0] == 1 and tmp['predict'].values[0] >= throsed1) or (tmp['model'
                                                                                            ].values[0] == 2 and
                                                                                            tmp['predict'].values[
                                                                                                0] >= throsed2):
                final_result = final_result.append(tmp)

        # 跟新阈值
        if final_result.shape[0] > 0:
            #         print(throsed)
            throsed1 = final_result.loc[final_result.model == 1]['predict'].min()
            throsed2 = final_result.loc[final_result.model == 2]['predict'].min()
        # 回溯前一天中未被召回的样本
        if last_curr1.shape[0] > 0:
            final_result = final_result.append(last_curr1.loc[last_curr1['predict'] >= throsed1])
            final_result = final_result.append(last_curr2.loc[last_curr2['predict'] >= throsed2])
        last_curr1 = model1_disk
        last_curr2 = model2_disk
    # 去除重复取的记录
    final_result.drop_duplicates(['serial_number', 'model', 'dt'], inplace=True)
    final_result.serial_number = final_result.serial_number.apply(lambda x: 'disk_' + str(x))
#-------------------------------------------------------------
    ##
    ###新策略评估1.3： 分model，取当天topN，之后去重
    val_x['predict'] = val_pred
    # test_8['predict']=test_pred
    sub = val_x[['model', 'serial_number', 'dt', 'predict', ]]
    sub['id'] = sub['serial_number'] * 10 + sub['model']
    dayss = sorted(sub['dt'].dt.day.unique())
    final_result = pd.DataFrame()
    select_data = {}
    last_curr = pd.DataFrame()
    throsed = 999
    # 一天天的数据预测
    # data6_model1_scale=np.round(np.array(data8_model1)/sum(data8_model1)*70)
    # data6_model2_scale=np.round(np.array(data8_model2)/sum(data8_model2)*70)
    model1_nn = sub.loc[sub.model == 1].shape[0]
    model2_nn = sub.loc[sub.model == 2].shape[0]
    for d in dayss:
        curr_data = sub.loc[sub['dt'].dt.day == d]
        model1_disk = curr_data.loc[curr_data.model == 1]
        model2_disk = curr_data.loc[curr_data.model == 2]
        model1_disk['rank'] = model1_disk['predict'].rank(ascending=False)
        model2_disk['rank'] = model2_disk['predict'].rank(ascending=False)
        nn1 = np.round((70 * model1_disk.shape[0] / model1_nn))
        nn2 = np.round((70 * model2_disk.shape[0] / model2_nn))
        print('nn1:{} nn2:{}'.format(nn1, nn2))
        tmp = pd.concat([model1_disk.loc[model1_disk['rank'] <= nn1], model2_disk.loc[model2_disk['rank'] <= nn2]],
                        axis=0)
        final_result = final_result.append(tmp)
        sub = sub.loc[~sub.id.isin(tmp.id)]
    # 去除重复取的记录
    final_result.drop_duplicates(['serial_number', 'model', 'dt'], inplace=True)
    final_result.serial_number = final_result.serial_number.apply(lambda x: 'disk_' + str(x))


    ###  上述方法运行其中一个，进行评估
    ## 评估f1
    # print(throsed)
    disk_len = len(final_result.drop_duplicates(['serial_number', 'model']))
    print('disk number:', disk_len)
    print('data len :', final_result.shape[0])
    a = outline_evalue(tag, final_result, mon=7)
    print(a['tag'].value_counts())