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
    data_45678=pd.read_pickle('../user_data/process_data/process_data_345678.pkl')
    smartcol = sorted([col for col in data_45678.columns if col not in ['dt', 'fault_time',
                                                                        'tag', 'label', 'gap_bad_day', 'manufacturer',
                                                                        'init_dt', 'init_smart_192raw', 'smart_4raw',
                                                                        'smart_195raw'
        , 'smart_12raw', 'smart_1_normalized', 'smart_190_normalized', 'times', 'serial_number', 'model', 'pred_tag']])
    # 'mean_diff_smart_1_normalized' ,,'mean_diff_smart_1raw',
    print(smartcol)
    print(len(smartcol))

    # 简单训练了tag标记的多分类模型，其实还可以细化做交叉验证，并用全量数据来提高准确度
    #这里只用4、5、6 月坏盘做训练
    train_x = data_45678.loc[data_45678.tag != 0] # 去除好盘
    val_x = train_x.loc[train_x['dt'].dt.month == 7]
    print(val_x.shape)
    val_y = val_x.tag
    train_x = train_x.loc[train_x['dt'].dt.month < 7]
    print(train_x.shape)
    train_y = train_x.tag
    print('================================= training validate ===============================================')
    fea_imp_list = []
    clf = LGBMClassifier(
        learning_rate=0.07,
        n_estimators=300,
        num_leaves=512,
        subsample=0.8,
        max_depth=-1,
        colsample_bytree=0.8,
        random_state=2018,
        is_unbalenced='True',
        objective='multiclass',
        #     metric=['binary_logloss']
    )  # 'binary_error','xentropy'
    print('************** training **************')

    clf = clf.fit(
        train_x[smartcol], train_y,
        eval_set=[(val_x[smartcol], val_y)],
        eval_metric=['multi_logloss'],  # ,'binary_error','xentropy'
        categorical_feature='auto',
        early_stopping_rounds=50,
        verbose=50
    )
    tag_pred = clf.predict_proba(data_45678[smartcol])
    fea_imp_list.append(clf.feature_importances_)

    # 保存模型
    with open('tag_model.pkl', 'wb') as pickle_file:
        pickle.dump(clf, pickle_file)
