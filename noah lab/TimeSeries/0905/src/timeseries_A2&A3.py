#encoding=utf-8
###A3预测A3



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
predict_month=['201506','201606']
###两条路径，一个行数,两个序列
train_year=6


def transform(df1,df2,df3):
    data_and_label=[]
    for i in range(0,172):
        data_and_label.append([])
        for j in range(1,len(df1.columns)-train_year):
            flag = True
            for k in range(train_year+1):               ##训练和标签都不包含预测月
                if str(df1.columns[j+k]) in predict_month:
                    flag = False
                    # continue
            if flag == True:
                temp=list(df1.iloc[i,j:(j+train_year)].values)
                temp.extend(list(df2.iloc[i, j:(j + train_year)].values))
                temp.extend([df3.iloc[i, j + train_year]])
                data_and_label[i].extend([temp])
                # data_and_label[i].extend([df2.iloc[i, j:(j + train_year)].values])
                # data_and_label[i].extend([df3.iloc[i,j + train_year]])
                # data_and_label[i].append([df1.iloc[i, j:(j + train_year)].values])
                # data_and_label[i].extend([df2.iloc[i, j:(j + train_year)].values])
                # data_and_label[i].extend([df3.iloc[i, j + train_year]])
    return data_and_label




oriData1 = pd.read_excel('../data/A1.xlsx', sheetname='Sheet1')
oriData1 = oriData1.fillna(0.0)
oriData2 = pd.read_excel('../data/A2.xlsx', sheetname='Sheet1')
oriData2 = oriData2.fillna(0.0)
oriData3 = pd.read_excel('../data/A3.xlsx', sheetname='Sheet1')
oriData3 = oriData3.fillna(0.0)
for predict_month in [['201506','201606'],['201512','201612']]:
    for train_year in [1,3,6]:
        print predict_month
        print 'train_year:\t' + str(train_year)
        train_and_label_data=[]
        train_and_label_data=transform(oriData1,oriData2,oriData3)
    ###对每个公司：
        for i in range(0,172):
            if algo == 'rf':
                model = RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=None, min_samples_split=2,
                                              max_features=0.5, n_jobs=-1)
            elif algo == 'gbrt':
                model = GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=1000, min_samples_split=2,
                                                  max_depth=5, max_features='sqrt', alpha=0.9)
            train_data = np.array(train_and_label_data)[i][:, :-1]
            label_data = np.array(train_and_label_data)[i][:, -1]
            model.fit(train_data,label_data)
            test_data1=[]

            ###18-24,26-32注意修改
            if predict_month==['201506','201606']:
                temp=list(oriData1.iloc[i,18-train_year:18].values)
                temp.extend(list(oriData2.iloc[i, 18 - train_year:18].values))
                test_data1.extend([temp])
                pred_data1=model.predict(test_data1)
                print '%f\t'% float(pred_data1),
                test_data2=[]
                # test_data2.extend(oriData.iloc[i, 31:32].values)
                # for i in range(5):
                #     temp=model.predict([test_data2])
                #     test_data2.append(float(temp))
                #     del test_data2[0]
                # print '%f'%temp
                test_data1 = []
                temp = list(oriData1.iloc[i, 30 - train_year:30].values)
                temp.extend(list(oriData2.iloc[i, 30 - train_year:30].values))
                test_data1.extend([temp])
                pred_data1 = model.predict(test_data1)
                print '%f\t' % pred_data1
            else:
                temp = list(oriData1.iloc[i, 24 - train_year:24].values)
                temp.extend(list(oriData2.iloc[i, 24 - train_year:24].values))
                test_data1.extend([temp])
                pred_data1 = model.predict(test_data1)
                print '%f\t' % pred_data1
                # test_data2 = []
                # test_data2.extend(oriData1.iloc[i, 32-train_year:32].values)
                # test_data2.extend(oriData2.iloc[i, 32 - train_year:32].values)
                # for i in range(5):
                #     temp=model.predict([test_data2])
                #     test_data2.append(float(temp))
                #     del test_data2[0]
                # print '%f'%temp



