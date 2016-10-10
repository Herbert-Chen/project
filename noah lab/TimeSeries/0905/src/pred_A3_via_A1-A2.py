# encoding=utf-8
###A1-A2预测A3



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
predict_month = '201506'
###两条路径，一个行数,两个序列
train_year = 1
var_name = 'A'
if var_name == 'A':
    total_colnum = 172
    method = 'pred_A3_via_A1-A2'
else:
    total_colnum = 112
    method = 'pred_B3_via_B1-B2'

###python3里面要修改
colnum=(int(predict_month)/100-2014)*12+int(predict_month)%100

def transform(df):
    data_and_label = []
    for i in range(0, total_colnum+1):
        data_and_label.append([])
        for j in range(1, len(df.columns) - train_year):
            flag = True
            for k in range(train_year + 1):  ##训练和标签都不包含预测月
                if str(df.columns[j + k]) == predict_month:
                    flag = False
                    # continue

            if flag == True:
                data_and_label[i].extend([df.iloc[i, j:(j + train_year + 1)].values])
            else:
                break
    return data_and_label

for var_name in ['A','B']:#['A','B']
    if var_name == 'A':
        total_colnum = 172
        method='pred_A3_via_A1-A2'
    else:
        total_colnum = 112
        method = 'pred_B3_via_B1-B2'


    oriData1 = pd.read_excel('../data/' + var_name + '1.xlsx', sheetname='Sheet1')
    oriData1 = oriData1.fillna(0.0)
    oriData2 = pd.read_excel('../data/' + var_name + '2.xlsx', sheetname='Sheet1')
    oriData2 = oriData2.fillna(0.0)
    oriData3 = pd.read_excel('../data/' + var_name + '3.xlsx', sheetname='Sheet1')
    oriData3 = oriData3.fillna(0.0)
    # for predict_month in ['201412','201506','201512','201606']:
    #     for train_year in [1,3,6]:
    #         print var_name
    #         print predict_month
    #         print train_year
    #         train_and_label_data1 = []
    #         train_and_label_data1 = transform(oriData1)
    #         train_and_label_data2 = []
    #         train_and_label_data2 = transform(oriData2)
    #
    #         ###对每个公司：
    #         for i in range(total_colnum, total_colnum+1):
    #             if algo == 'rf':
    #                 model = RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=None, min_samples_split=2,
    #                                               max_features=0.5, n_jobs=-1)
    #             elif algo == 'gbrt':
    #                 model = GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=1000, min_samples_split=2,
    #                                                   max_depth=5, max_features='sqrt', alpha=0.9)
    #             train_data = np.array(train_and_label_data1)[i][:, :-1]
    #             label_data = np.array(train_and_label_data1)[i][:, -1]
    #             model.fit(train_data, label_data)
    #             colnum = (int(predict_month) / 100 - 2014) * 12 + int(predict_month) % 100
    #             test_data1 = []
    #
    #             ###18-24,26-32注意修改
    #             test_data1.extend([oriData1.iloc[i, colnum-train_year:colnum].values])
    #             pred_data1 = model.predict(test_data1)
    #
    #
    #             if algo == 'rf':
    #                 model = RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=None, min_samples_split=2,
    #                                               max_features=0.5, n_jobs=-1)
    #             elif algo == 'gbrt':
    #                 model = GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=1000, min_samples_split=2,
    #                                                   max_depth=5, max_features='sqrt', alpha=0.9)
    #             train_data = np.array(train_and_label_data2)[i][:, :-1]
    #             label_data = np.array(train_and_label_data2)[i][:, -1]
    #             model.fit(train_data, label_data)
    #             colnum = (int(predict_month) / 100 - 2014) * 12 + int(predict_month) % 100
    #             test_data2 = []
    #
    #             ###18-24,26-32注意修改
    #             test_data2.extend([oriData2.iloc[i, colnum-train_year:colnum].values])
    #             pred_data2 = model.predict(test_data2)
    #             print '%f\t%f\t%f\t' % (float(pred_data1),float(pred_data2),float(pred_data1-pred_data2))
    #         print '\n'

    predict_month='201612'
    for train_year in [1, 3, 6]:
        train_and_label_data1 = []
        train_and_label_data1 = transform(oriData1)
        train_and_label_data2 = []
        train_and_label_data2 = transform(oriData2)
        print var_name
        print predict_month
        print train_year


        ###对每个公司：
        for i in range(total_colnum, total_colnum+1):
            if algo == 'rf':
                model = RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=None, min_samples_split=2,
                                              max_features=0.5, n_jobs=-1)
            elif algo == 'gbrt':
                model = GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=1000,
                                                  min_samples_split=2,
                                                  max_depth=5, max_features='sqrt', alpha=0.9)
            train_data = np.array(train_and_label_data1)[i][:, :-1]
            label_data = np.array(train_and_label_data1)[i][:, -1]
            model.fit(train_data, label_data)
            colnum = (int(predict_month) / 100 - 2014) * 12 + int(predict_month) % 100
            ###18-24,26-32注意修改
            test_data1=[]
            test_data1.extend(list(oriData1.iloc[i, 32-train_year:32].values))
            A1_2016=[]

            #迭代次数
            for times in range(5):
                temp=model.predict([test_data1])
                test_data1.append(float(temp))
                A1_2016.append(float(temp))
                del test_data1[0]
            pred_A1=temp

            if algo == 'rf':
                model = RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=None,
                                              min_samples_split=2,
                                              max_features=0.5, n_jobs=-1)
            elif algo == 'gbrt':
                model = GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=1000,
                                                  min_samples_split=2,
                                                  max_depth=5, max_features='sqrt', alpha=0.9)
            train_data = np.array(train_and_label_data2)[i][:, :-1]
            label_data = np.array(train_and_label_data2)[i][:, -1]
            model.fit(train_data, label_data)

            ###18-24,26-32注意修改
            test_data2 = []
            test_data2.extend(list(oriData2.iloc[i, 32 - train_year:32].values))
            A2_2016=[]
            # 迭代次数
            for times in range(5):
                temp = model.predict([test_data2])
                test_data2.append(float(temp))
                A2_2016.append(float(temp))
                del test_data2[0]
            pred_A2=temp
            #print '%f\t%f\t%f\t'%(pred_A1,pred_A2,pred_A1-pred_A2)
        data_2016 = []
        data_2016.extend(oriData3.iloc[i, 32 - 7:32].values)
        temp = [A1_2016[i] - A2_2016[i] for i in range(len(A1_2016))]
        data_2016.extend(temp)
        for i in range(len(data_2016)):
            print str(201600 + 1 + i) + ':\t' + str(data_2016[i])

        print '\n'



        ###作图
        year_2016 = [x + 201600 for x in range(1, 13)]
        pyplot.figure()
        pyplot.plot(year_2016, data_2016)
        pyplot.xlabel('year')
        pyplot.ylabel('data')
        pyplot.title(method + '_%d' % train_year)
        pyplot.savefig('../result/0913' + u'合并抵消' + '/' + method + '_%d.jpg' % train_year)



