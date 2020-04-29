import pandas as pd
import gc
import pickle
from lightgbm.sklearn import LGBMClassifier,LGBMRegressor
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import auc

if __name__ == '__main__':
    data_45678 = pd.read_pickle('../user_data/process_data/process_data_345678.pkl')
    smartcol = sorted([col for col in data_45678.columns if col not in ['dt', 'fault_time',
                                                                        'tag', 'label', 'gap_bad_day', 'manufacturer',
                                                                        'init_dt', 'init_smart_192raw', 'smart_4raw',
                                                                        'smart_195raw'
        , 'smart_12raw', 'smart_1_normalized', 'smart_190_normalized', 'times', 'serial_number', 'model', 'pred_tag']])
    # 'mean_diff_smart_1_normalized' ,,'mean_diff_smart_1raw',
    print(smartcol)
    print(len(smartcol))

    # -------------分训练集--4 - 5
    train_x = data_45678.loc[(data_45678['dt'].dt.month == 5) | (data_45678['dt'].dt.month == 4)]  #
    print(len(train_x))
    # train_x.drop(train_x.loc[train_x['gap_bad_day']<4].index,axis=0,inplace=True)
    # train_x.reset_index(drop=True,inplace=True)
    # print(len(train_x))
    train_y = train_x.loc[:, 'label']  #
    # 6
    data_6 = data_45678.loc[(data_45678['dt'].dt.month == 6)]

    # 7
    val_x = data_45678.loc[data_45678['dt'].dt.month == 7]
    val_y = data_45678.loc[data_45678['dt'].dt.month == 7, 'label']

    test_8 = data_45678.loc[(data_45678['dt'].dt.month == 8)]
    # del data_45678
    gc.collect()

### 可选
## 简单的随机搜索出一些比较好的参数
    clf = LGBMRegressor(
        learning_rate=0.001,
        n_estimators=100,
        num_leaves=156,
        subsample=0.8,
        max_depth=-1,
        colsample_bytree=0.8,
        random_state=2018,  # 2019
        is_unbalenced='True',
        metric=['auc', 'mse'],
        #     objective='regression_l1'#'poisson'
    )
    param_grid = {'learning_rate': [0.001 + i * 0.001 for i in range(0, 10)],
                  'num_leaves': [256 + i for i in range(0, 300, 32)],
                  'lambda_l1': [0.1 + 0.1 * i for i in range(20)],
                  'lambda_l2': [1 + i * 0.5 for i in range(20)],
                  'colsample_bytree': [0.8, 0.75, 0.6, 0.85, 0.5, 0.9],
                  'subsample': [0.75, 0.6, 0.7, 0.8, 0.9, 0.85]}
    rand = RandomizedSearchCV(clf, param_grid, cv=2, scoring='neg_mean_squared_error', n_iter=15, random_state=2020)

    print('随机搜索-度量记录：', rand.cv_results_)  # 包含每次训练的相关信息
    print('随机搜索-最佳度量值:', rand.best_score_)  # 获取最佳度量值
    print('随机搜索-最佳参数：', rand.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
    print('随机搜索-最佳模型：', rand.best_estimator_)  # 获取最佳度量时的分类器模型

    # 加载tag_model
    with open('tag_model.pkl', 'rb') as f:
        tag_model=pickle.load(f)

    train_tag = np.argmax(tag_model.predict_proba(train_x[smartcol]), axis=1)
    print('value')
    val_tag = np.argmax(tag_model.predict_proba(val_x[smartcol]), axis=1)
    data_6tag=np.argmax(tag_model.predict_proba(data_6[smartcol]), axis=1)
    train_x['tag_predict'] = train_tag
    val_x['tag_predict'] = val_tag
    data_6['tag_predict'] = data_6tag
    # 添加特征
    smartcol += ['tag_predict']
    ##---- 通过搜索出的参数，训练了两个不同参数的模型
    fea_imp_list = []
    val_pred = []
    best_iteration_list = []
    for p in [(512, 0.007, 0.6, 3, 0.6, 0.8), (476, 0.001, 0.85, 7, 1.3, 0.8)]:
        num_leaves, learning_rate, feature_fraction, lambda_l2, lambda_l1, subsample = p
        param = {
            'num_leaves': num_leaves,
            'objective': 'regression',  # regression_l2
            'max_depth': -1,
            'learning_rate': learning_rate,
            'boosting': 'gbdt',
            'feature_fraction': feature_fraction,
            'bagging_fraction': subsample,
            'bagging_freq': 2,
            'metric': ['auc', 'mse'],
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'nthread': -1,
            'verbosity': -1,
        }
        lgb_train = lgb.Dataset(train_x[smartcol], train_x['label'])
        lgb_val = lgb.Dataset(val_x[smartcol], val_x['label'])
        num_round = 300
        clf = lgb.train(param, lgb_train, num_round, valid_sets=[lgb_val],
                        verbose_eval=100, early_stopping_rounds=50, )
        fea_imp_list.append(clf.feature_importance())
        #     best_rounds=clf.best_iteration
        val_pred.append(clf.predict(val_x[smartcol], num_iteration=clf.best_iteration))
        best_iteration_list.append(clf.best_iteration) # 本来是想使用加权系数，但是两个模型的话训练轮次
        #差不多，所以最后直接用了平均
        ###----------------------------------------------------------
#--------------------------------------------------------------
        #### 用5 6 月训练模型
        train_x=pd.concat([data_6,val_x],axis=0)
        train_y=train_x['label']

        for index, p in enumerate([(512,0.007,0.6,3,0.6,0.8),(476,0.001,0.85,7,1.3,0.8)]):
            num_leaves, learning_rate, feature_fraction, lambda_l2, lambda_l1, subsample = p
            param = {
                'num_leaves': num_leaves,
                'objective': 'regression',  # regression_l2
                'max_depth': -1,
                'learning_rate': learning_rate,
                'boosting': 'gbdt',
                'feature_fraction': feature_fraction,
                'bagging_fraction': subsample,
                'bagging_freq': 2,
                'metric': ['auc', 'mse'],
                'lambda_l1': lambda_l1,
                'lambda_l2': lambda_l2,
                'nthread': -1,
                'verbosity': -1,
            }
            lgb_train = lgb.Dataset(train_x[smartcol], train_y)
            #     num_round =best_iteration_list[index]
            num_round = best_iteration_list[index]
            clf = lgb.train(param, lgb_train, num_round, valid_sets=[lgb_train],
                            verbose_eval=50)
            fea_imp_list.append(clf.feature_importance())
            with open('ensenble_lgb{}_{}.pkl'.format(index,num_round),'wb') as f:
                pickle.dump(clf,f)

