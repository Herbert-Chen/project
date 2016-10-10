# encoding=utf-8
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
predict_month = '201506'
###两条路径，一个行数,两个序列
train_year = 1
var_name = 'A'
if var_name == 'A':
    total_colnum = 172
    method = 'A3_pred_A3'
else:
    total_colnum = 112
    method = 'B3_pred_B3'

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

for var_name in ['A','B']:
    if var_name == 'A':
        total_colnum = 172
        method='A3_pred_A3'
    else:
        total_colnum = 112
        method = 'B3_pred_B3'


    oriData = pd.read_excel('../data/' + var_name + '3_2.xlsx', sheetname='Sheet1')
    oriData = oriData.fillna(0.0)
    # for predict_month in ['201412','201506','201512','201606']:
    #     for train_year in [1,3,6]:
    #         print var_name
    #         print predict_month
    #         print train_year
    #         train_and_label_data = []
    #         train_and_label_data = transform(oriData)
    #
    #         ###对每个公司：
    #         for i in range(0, total_colnum+1):
    #             if algo == 'rf':
    #                 model = RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=None, min_samples_split=2,
    #                                               max_features=0.5, n_jobs=-1)
    #             elif algo == 'gbrt':
    #                 model = GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=1000, min_samples_split=2,
    #                                                   max_depth=5, max_features='sqrt', alpha=0.9)
    #             train_data = np.array(train_and_label_data)[i][:, :-1]
    #             label_data = np.array(train_and_label_data)[i][:, -1]
    #             model.fit(train_data, label_data)
    #             colnum = (int(predict_month) / 100 - 2014) * 12 + int(predict_month) % 100 +1   ###
    #             test_data1 = []
    #
    #             ###18-24,26-32注意修改
    #             test_data1.extend([oriData.iloc[i, colnum-train_year:colnum].values])
    #             pred_data1 = model.predict(test_data1)
    #             print '%f\t' % float(pred_data1)
    #         print '\n'

    predict_month='201612'
    for train_year in [1, 3, 6]:
        train_and_label_data = []
        train_and_label_data = transform(oriData)
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
            train_data = np.array(train_and_label_data)[i][:, :-1]
            label_data = np.array(train_and_label_data)[i][:, -1]
            model.fit(train_data, label_data)
            colnum = (int(predict_month) / 100 - 2014) * 12 + int(predict_month) % 100  +1      ###A3_2,201312新增
            ###18-24,26-32注意修改
            test_data2=[]
            test_data2.extend(oriData.iloc[i, 34-train_year:34].values)

            # ###作图
            data_2016=[]
            data_2016.extend(oriData.iloc[i, 34-8:34].values)
            #迭代次数
            for i in range(4):
                temp=model.predict([test_data2])
                test_data2.append(float(temp))
                data_2016.append(float(temp))
                del test_data2[0]
            # print '%f'%temp
        for i in range(len(data_2016)):
            print str(201600+1+i)+':\t'+str(data_2016[i])
        print '\n'

        # ###作图
        year_2016=[x+201600 for x in range(1,13)]
        pyplot.figure()
        pyplot.plot(year_2016,data_2016)
        pyplot.xlabel('year')
        pyplot.ylabel('data')
        pyplot.title(var_name+'3_pred_'+var_name+'3_%d'%train_year)
        pyplot.savefig('../result/0921'+u'合并抵消'+'/'+var_name+'3_pred_'+var_name+'3_%d.jpg'%train_year)

        # data_2015 = []
        # data_2015.extend(list(oriData.iloc[172, 13:25].values))
        # year_2015=[x+201500 for x in range(1,13)]
        # pyplot.figure()
        # pyplot.plot(year_2015, data_2015)
        # pyplot.xlabel('year')
        # pyplot.ylabel('data')
        # pyplot.title('year 2015')
        # pyplot.savefig('../result/0913' + u'合并抵消' + '/year 2015')
        # for i in range(len(data_2015)):
        #     print str(201500+1+i)+':\t'+str(data_2015[i])
        #
        # data_2014 = []
        # data_2014.extend(list(oriData.iloc[172, 1:13].values))
        # year_2014 = [x + 201400 for x in range(1, 13)]
        # pyplot.figure()
        # pyplot.plot(year_2014, data_2014)
        # pyplot.xlabel('year')
        # pyplot.ylabel('data')
        # pyplot.title('year 2014')
        # pyplot.savefig('../result/0913' + u'合并抵消' + '/year 2014')
        # for i in range(len(data_2014)):
        #     print str(201400 + 1 + i) + ':\t' + str(data_2014[i])
