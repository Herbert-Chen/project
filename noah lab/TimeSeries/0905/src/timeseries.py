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
train_year=1
var_name='A'
colnum=172

def transform(df):
    data_and_label=[]
    for i in range(0,colnum):
        data_and_label.append([])
        for j in range(1,len(df.columns)-train_year):
            flag = True
            for k in range(train_year+1):               ##训练和标签都不包含预测月
                if str(df.columns[j+k]) in predict_month:
                    flag = False
                    # continue

            if flag == True:
                data_and_label[i].extend([df.iloc[i,j:(j+train_year+1)].values])
            else:
                break
    return data_and_label




oriData = pd.read_excel('../data/'+var_name+'3.xlsx', sheetname='Sheet1')
oriData = oriData.fillna(0.0)
train_and_label_data=[]
train_and_label_data=transform(oriData)





###对每个公司：
for i in range(0,colnum):
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
    test_data1.extend([oriData.iloc[i,12:18].values])
    pred_data1=model.predict(test_data1)
    print '%f\t'%pred_data1,
    test_data2=[]
    # test_data2.extend(oriData.iloc[i, 31:32].values)
    # for i in range(5):
    #     temp=model.predict([test_data2])
    #     test_data2.append(float(temp))
    #     del test_data2[0]
    # print '%f'%temp
    test_data1 = []
    test_data1.extend([oriData.iloc[i, 24:30].values])
    pred_data1 = model.predict(test_data1)
    print '%f\t' % pred_data1


