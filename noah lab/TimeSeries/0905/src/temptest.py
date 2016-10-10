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
else:total_colnum=112

###python3里面要修改
colnum=(int(predict_month)/100-2014)*12+int(predict_month)%100

def transform(df):
    data_and_label = []
    for i in range(0, total_colnum):
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

for var_name in ['A']:
    if var_name == 'A':
        total_colnum = 172
    else:
        total_colnum = 112


    oriData = pd.read_excel('../data/' + var_name + '3.xlsx', sheetname='Sheet1')
    oriData = oriData.fillna(0.0)
    for predict_month in ['201412']:
        for train_year in [1,3,6]:
            print var_name
            print predict_month
            print train_year
            train_and_label_data = []
            train_and_label_data = transform(oriData)

            ###对每个公司：
            for i in range(0, total_colnum):
                if algo == 'rf':
                    model = RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=None, min_samples_split=2,
                                                  max_features=0.5, n_jobs=-1)
                elif algo == 'gbrt':
                    model = GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=1000, min_samples_split=2,
                                                      max_depth=5, max_features='sqrt', alpha=0.9)
                train_data = np.array(train_and_label_data)[i][:, :-1]
                label_data = np.array(train_and_label_data)[i][:, -1]
                model.fit(train_data, label_data)
                colnum = (int(predict_month) / 100 - 2014) * 12 + int(predict_month) % 100
                test_data1 = []

                ###18-24,26-32注意修改
                test_data1.extend([oriData.iloc[i, colnum-train_year:colnum].values])
                pred_data1 = model.predict(test_data1)
                print '%f\t' % float(pred_data1)
            print '\n'

