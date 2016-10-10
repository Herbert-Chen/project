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


algo = 'gbrt'

oriData13 = pd.read_excel('../data/cx2013.xlsx', sheetname='Sheet1')
oriData13 = oriData13.fillna(0.0)
oriData14 = pd.read_excel('../data/cx2014.xlsx', sheetname='Sheet1')
oriData14 = oriData14.fillna(0.0)
oriData15 = pd.read_excel('../data/cx2015.xlsx', sheetname='Sheet1')
oriData15 = oriData15.fillna(0.0)

for i in range(1,193):
    dataset_i_13=oriData13[oriData13['unit']==i]
    dataset_i_14 = oriData14[oriData14['unit'] == i]
    dataset_i_15 = oriData15[oriData15['unit'] == i]
    dataset_i_14=dataset_i_14[dataset_i_14['rep'].isin(dataset_i_13['rep'])]
    dataset_i_15 = dataset_i_15[dataset_i_15['rep'].isin(dataset_i_13['rep'])]

    dataset_i_13 = dataset_i_13[dataset_i_13['rep'].isin(dataset_i_14['rep'])]
    dataset_i_15 = dataset_i_15[dataset_i_15['rep'].isin(dataset_i_14['rep'])]

    dataset_i_14 = dataset_i_14[dataset_i_14['rep'].isin(dataset_i_15['rep'])]
    dataset_i_13 = dataset_i_13[dataset_i_13['rep'].isin(dataset_i_15['rep'])]


    dataset_i_13=np.array(dataset_i_13)
    dataset_i_14 = np.array(dataset_i_14)
    dataset_i_15 = np.array(dataset_i_15)
    dataset=np.column_stack((dataset_i_13,dataset_i_14[:,3:15],dataset_i_15[:,3:15]))
    train_data=[]
    train_label=[]
    for j in range(23):
        train_data.append(dataset[:,3+j])
        train_label.append(dataset[dataset.shape[0]-1-2,3+j+1])

    if algo == 'rf':
        model = RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=None, min_samples_split=2,
                                      max_features=0.5, n_jobs=-1)
    elif algo == 'gbrt':
        model = GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=1000, min_samples_split=2,
                                          max_depth=5, max_features='sqrt', alpha=0.9)
    model.fit(train_data, train_label)
    for k in range(12):
        print str(dataset[dataset.shape[0] - 1 - 2, 27+k])+'\t',
    print '\t\t',

    test_data=[dataset[:,3+23]]
    for k in range(12):
        pred_cx=model.predict(test_data)
        print str(pred_cx[0])+'\t',
        test_data=[dataset[:,3+23+1+k]]
        test_data[0][dataset.shape[0]-1-2]=pred_cx

    print
