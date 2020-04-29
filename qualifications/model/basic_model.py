import pandas as pd
import gc
import pickle
from lightgbm.sklearn import LGBMClassifier
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
### 用于线下f1评估，仿照赛方规则，label是整月的所有数据，只能去7月以前的，才有完整标签
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

    data_345678=pd.read_pickle('../user_data/process_data/process_data_345678.pkl')
    print('读取tag-----------------------')
    tag = pd.read_csv('../data/round1_train/fault_tag.csv')
    tag['fault_time'] = pd.to_datetime(tag['fault_time'])
    ###tag表里面有的硬盘同一天发生几种故障
    tag['tag'] = tag['tag'].astype(str)
    # 去掉4 5 6 标签的样本
    part_tag = tag[(tag['tag'] != '4') & (tag['tag'] != '6') & (tag['tag'] != '5')]  # 这份标记用于训练集
    tag = tag.groupby(['serial_number', 'fault_time', 'model'])['tag'].apply(lambda x: '|'.join(x)).reset_index()
    ## 训练特征列表
    smartcol=[col for col in data_345678.columns if col not in ['dt','fault_time',
        'tag','label','gap_bad_day','manufacturer','init_dt']]
    print('training feature:',smartcol)
    print(len(smartcol))
    ##-------------分训练集--
    train_x=data_345678.loc[data_345678['dt'].dt.month<5]
    train_y=data_345678.loc[data_345678['dt'].dt.month<5,'label']

    data_5=data_345678.loc[data_345678['dt'].dt.month==5]

    val_x=data_345678.loc[data_345678['dt'].dt.month==6]
    val_y=data_345678.loc[data_345678['dt'].dt.month==6,'label']

    test_df=data_345678.loc[data_345678['dt'].dt.month>7]
    del data_345678
    gc.collect()

    #-------------train-val------------------
    print('================================= training validate ===============================================')
    fea_imp_list = []
    clf = LGBMClassifier(
        learning_rate=0.001,
        n_estimators=400,
        num_leaves=156,
        subsample=0.8,
        max_depth=7,
        colsample_bytree=0.8,
        random_state=2019,
        is_unbalenced='True',
        metric=['auc'])
    print('************** training **************')
    clf.fit(
        train_x[smartcol], train_y,
        eval_set=[(val_x[smartcol], val_y)],
        eval_metric=['auc'],
        categorical_feature='auto',
        early_stopping_rounds=50,
        verbose=50
    )
    print('************** validate predict **************')
    best_rounds = clf.best_iteration_
    best_auc = clf.best_score_['valid_0']['auc']
    val_pred = clf.predict_proba(val_x[smartcol])[:, 1]
    fea_imp_list.append(clf.feature_importances_)
    ##--------评估 val--------
    val_x['predict'] = val_pred
    sub = val_x[['model', 'serial_number', 'dt', 'predict']]
    sub = sub.sort_values('predict', ascending=False)
    sub = sub.drop_duplicates(['serial_number', 'model'])
    # sub['label']=[1 if x >=throse+0.002  else 0 for x in sub['predict']]
    sub['label'] = sub['predict'].rank()
    sub['label'] = (sub['label'] >= sub.shape[0] * 0.9989).astype(int)
    sub = sub.loc[sub.label == 1]
    print(len(sub))
    nout = len(sub)
    thros = sub['predict'].min()
    sub.serial_number = sub.serial_number.apply(lambda x: 'disk_' + str(x))
    print('the throsed:',sub['predict'].min())
    ## 评估f1
    a = outline_evalue(tag, sub, mon=6)
    print(a['tag'].value_counts())
    ## 得到特征重要性
    fea_imp_dict = dict(zip(smartcol, np.mean(fea_imp_list, axis=0)))
    fea_imp_item = sorted(fea_imp_dict.items(), key=lambda x: x[1], reverse=True)
    for f, imp in fea_imp_item:
        print('{} = {}'.format(f, imp))
    np.save('../user_data/tmp_data/import_feature.npy', fea_imp_item)  # 保存特征重要性  'mean_diff_smart_7raw','smart_195_normalized'
    ## 重新组合训练集
    train_x = pd.concat([data_5, val_x], axis=0).reset_index(drop=True)
    train_y = pd.concat([data_5['label'], val_x['label']], axis=0).reset_index(drop=True)
    del data_5
    print(
        '=============================================== training predict ===============================================')
    fea_imp_list = []
    clf = LGBMClassifier(
        learning_rate=0.001,
        n_estimators=best_rounds,
        num_leaves=156,
        subsample=0.8,
        max_depth=7,
        colsample_bytree=0.8,
        random_state=2019, is_unbalenced='True',
    )
    print('************** training **************')
    clf.fit(
        train_x[smartcol], train_y,
        eval_set=[(train_x[smartcol], train_y)],
        categorical_feature='auto',
        verbose=50)
    print('save model ./lgb_model.pkl ------')
    pickle.dump(clf,'./lgb_model.pkl')
    test_df['predict'] = clf.predict_proba(test_df[smartcol])[:, 1]

    fea_imp_list.append(clf.feature_importances_)
    fea_imp_dict = dict(zip(smartcol, np.mean(fea_imp_list, axis=0)))
    fea_imp_item = sorted(fea_imp_dict.items(), key=lambda x: x[1], reverse=True)
    for f, imp in fea_imp_item:
        print('{} = {}'.format(f, imp))
    np.save('../user_data/tmp_data/import_feature.npy', fea_imp_item)  # 保存特征重要性
    #--------- inference---------
    sub = test_df[['manufacturer', 'model', 'serial_number', 'dt', 'predict']]
    sub = sub[(sub['dt'].dt.month == 8) & (sub['model'] == 2)]

    sub = sub.sort_values('predict', ascending=False)
    sub = sub.drop_duplicates(['serial_number', 'model'])
    sub['label']=(sub['predict']>=thros).astype(int)
    # sub['label'] = sub['predict'].rank()
    # sub['label'] = (sub['label'] >= sub.shape[0] * 0.992).astype(int)
    sub = sub.loc[sub.label == 1]
    nout = len(sub)
    thros = sub['predict'].min()
    print(nout)
    print(thros)
    sub.serial_number = sub.serial_number.apply(lambda x: 'disk_' + str(x))
    sub[['manufacturer', 'model', 'serial_number', 'dt']].to_csv(
        "../prediction_result/predictions.csv".format(thros, nout), index=False, header=None)
