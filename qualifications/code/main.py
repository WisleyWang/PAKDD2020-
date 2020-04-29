import pickle
import pandas as pd
import numpy as np
import gc
from deepctr.models import DCN,DeepFM,xdeepfm,DIN
if __name__ == '__main__':
    # 加载模型
    clf=pickle.load('../model/lgb_model.pkl')
    # 读取数据
    data_345678=pd.read_pickle('../user_data/process_data/process_data_345678.pkl')
    # 获得测试集
    test_df = data_345678.loc[data_345678['dt'].dt.month > 7]
    ## 训练特征列表
    smartcol = [col for col in data_345678.columns if col not in ['dt', 'fault_time',
                                                                  'tag', 'label', 'gap_bad_day', 'manufacturer',
                                                                  'init_dt']]
    print('training feature:', smartcol)
    print(len(smartcol))
    del data_345678
    gc.collect()
    ### 预测
    test_df['predict'] = clf.predict_proba(test_df[smartcol])[:, 1]
    sub = test_df[['manufacturer', 'model', 'serial_number', 'dt', 'predict']]
    sub = sub[(sub['dt'].dt.month == 8) & (sub['model'] == 2)]

    sub = sub.sort_values('predict', ascending=False)
    sub = sub.drop_duplicates(['serial_number', 'model'])
    # sub['label']=(sub['predict']>=thros).astype(int)
    sub['label'] = sub['predict'].rank()
    sub['label'] = (sub['label'] >= sub.shape[0] * 0.992).astype(int)
    sub = sub.loc[sub.label == 1]
    nout = len(sub)
    thros = sub['predict'].min()
    print('预测数目:',nout)
    print('预测阈值',thros)
    sub.serial_number = sub.serial_number.apply(lambda x: 'disk_' + str(x))
    print("save predict to '../prediction_result/predictions.csv'")
    sub[['manufacturer', 'model', 'serial_number', 'dt']].to_csv(
        "../prediction_result/predictions.csv".format(thros, nout), index=False, header=None)
