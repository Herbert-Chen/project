#encoding=utf-8
###依赖包


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from sklearn import cross_validation
# from plot import ArkPlot
import sys
from scipy.stats import pearsonr
from matplotlib import pyplot
reload(sys)
sys.setdefaultencoding('utf-8')

elem=['AP','AQ','AR','AT','AU','AV','AW','AY','AZ','BA','BB','BC','BD','BX','BY','CA','CD','CE','CF','CG','CH','CO','CN','CT','CV','CX']
algo = 'gbrt'

oriData13 = pd.read_excel('../data/cx2013.xlsx', sheetname='Sheet1')
oriData13 = oriData13.fillna(0.0)
oriData14 = pd.read_excel('../data/cx2014.xlsx', sheetname='Sheet1')
oriData14 = oriData14.fillna(0.0)
oriData15 = pd.read_excel('../data/cx2015.xlsx', sheetname='Sheet1')
oriData15 = oriData15.fillna(0.0)
oriData16 = pd.read_excel('../data/cx2016.xlsx', sheetname='Sheet1')
oriData16 = oriData16.fillna(0.0)

for i in range(1,193):
    dataset_i_13=oriData13[oriData13['unit']==i]
    dataset_i_14 = oriData14[oriData14['unit'] == i]
    dataset_i_15 = oriData15[oriData15['unit'] == i]
    dataset_i_16 = oriData16[oriData16['unit'] == i]

    reduced_dataset_i_13 = dataset_i_13[dataset_i_13['rep'].isin(elem)]
    reduced_dataset_i_14 = dataset_i_14[dataset_i_14['rep'].isin(elem)]
    reduced_dataset_i_15 = dataset_i_15[dataset_i_15['rep'].isin(elem)]
    reduced_dataset_i_16 = dataset_i_16[dataset_i_16['rep'].isin(elem)]

    reduced_dataset_i_14 = reduced_dataset_i_14[reduced_dataset_i_14['rep'].isin(reduced_dataset_i_13['rep'])]
    reduced_dataset_i_15 = reduced_dataset_i_15[reduced_dataset_i_15['rep'].isin(reduced_dataset_i_13['rep'])]
    reduced_dataset_i_16 = reduced_dataset_i_16[reduced_dataset_i_16['rep'].isin(reduced_dataset_i_13['rep'])]

    reduced_dataset_i_13 = reduced_dataset_i_13[reduced_dataset_i_13['rep'].isin(reduced_dataset_i_14['rep'])]
    reduced_dataset_i_15 = reduced_dataset_i_15[reduced_dataset_i_15['rep'].isin(reduced_dataset_i_14['rep'])]
    reduced_dataset_i_16 = reduced_dataset_i_16[reduced_dataset_i_16['rep'].isin(reduced_dataset_i_14['rep'])]

    reduced_dataset_i_14 = reduced_dataset_i_14[reduced_dataset_i_14['rep'].isin(reduced_dataset_i_15['rep'])]
    reduced_dataset_i_13 = reduced_dataset_i_13[reduced_dataset_i_13['rep'].isin(reduced_dataset_i_15['rep'])]
    reduced_dataset_i_16 = reduced_dataset_i_16[reduced_dataset_i_16['rep'].isin(reduced_dataset_i_15['rep'])]

    reduced_dataset_i_13 = reduced_dataset_i_13[reduced_dataset_i_13['rep'].isin(reduced_dataset_i_16['rep'])]
    reduced_dataset_i_14 = reduced_dataset_i_14[reduced_dataset_i_14['rep'].isin(reduced_dataset_i_16['rep'])]
    reduced_dataset_i_15 = reduced_dataset_i_15[reduced_dataset_i_15['rep'].isin(reduced_dataset_i_16['rep'])]


    reduced_array_i_13 = np.array(reduced_dataset_i_13)
    reduced_array_i_14 = np.array(reduced_dataset_i_14)
    reduced_array_i_15 = np.array(reduced_dataset_i_15)
    reduced_array_i_16 = np.array(reduced_dataset_i_16)
    reduced_array=np.column_stack((reduced_array_i_13[:,:3],reduced_array_i_14[:,2],reduced_array_i_15[:,2],reduced_array_i_16[:,2],reduced_array_i_13[:,3:15],reduced_array_i_14[:,3:15],reduced_array_i_15[:,3:15]))
    train_data=[]
    train_label=[]
    for j in range(23):
        temp=list(reduced_array[:,6+j])
        temp.extend(list(reduced_array[:, 2 + j / 12]))
        temp.extend(list(reduced_array[:,3+j/12]))
        train_data.append(temp)
        train_label.append(reduced_array[-1,6+j+1])
    # train_data=np.array(train_data)
    if algo == 'rf':
        model = RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=None, min_samples_split=2,
                                      max_features=0.5, n_jobs=-1)
    elif algo == 'gbrt':
        model = GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=1000, min_samples_split=2,
                                          max_depth=5, max_features='sqrt', alpha=0.9)
    model.fit(train_data, train_label)
    for k in range(12):
        print str(reduced_array[ - 1, 6+24+k])+'\t',
    print '\t\t',

    test_data=list(reduced_array[:,6+23])
    test_data.extend(reduced_array[:, 3])
    test_data.extend(reduced_array[:,4])
    for k in range(12):
        pred_cx=model.predict([test_data])
        print str(pred_cx[0])+'\t',
        test_data=list(reduced_array[:,6+23+1+k])
        test_data[-1]=pred_cx
        test_data.extend(reduced_array[:, 4])
        test_data.extend(reduced_array[:,5 ])

    print
