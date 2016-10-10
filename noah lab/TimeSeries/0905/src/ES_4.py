# encoding=utf-8
###(Exponential Smoothing)



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
import random

reload(sys)
sys.setdefaultencoding('utf-8')



predict_month = '201609'
###两条路径，一个行数,两个序列
train_year = 6
var_name = 'A'
if var_name == 'A':
    total_colnum = 172
else:total_colnum=112

###python3里面要修改
colnum=(int(predict_month)/100-2014)*12+int(predict_month)%100

def transform_period(df):
    data_and_label = []
    for i in range(0, total_colnum+1):
        data_and_label.append([])
        for j in range(1, len(df.columns) - train_year,train_year):
            flag = True
            for k in range(train_year + 1+1):  ##首尾分别加1，首做基准尾做预测对比
                if str(df.columns[j + k]) == predict_month:
                    flag = False
                    # continue

            if flag == True:
                data_and_label[i].extend([list(df.iloc[i, j:(j + train_year + 1+1)].values)])##首尾分别加1，首做基准尾做预测对比
            else:
                break
    return data_and_label
def transform(df):
    data_and_label = []
    for i in range(0, total_colnum+1):
        data_and_label.append([])
        for j in range(1, len(df.columns) - train_year-2):
            flag = True
            for k in range(train_year + 1+1):  ##首尾分别加1，首做基准尾做预测对比
                if str(df.columns[j + k]) == predict_month:
                    flag = False
                    # continue

            if flag == True:
                data_and_label[i].extend([list(df.iloc[i, j:(j + train_year + 1+1)].values)])##首尾分别加1，首做基准尾做预测对比
            else:
                break
    return data_and_label
###输入实际数值和迭代次数，得到平滑值
def ES(datalist,n,alpha):
    ESdata=[]
    weight_list=[]
    ESdata.append(datalist[0])

    for times in range(1,min(n,len(datalist))):
        ESdata.append(alpha*datalist[times]+(1-alpha)*ESdata[times-1])
    if n > len(datalist):
        for times in range(len(datalist), n):
            ####此处可能有问题
            ESdata.append(alpha * ESdata[times-1] + (1 - alpha) * ESdata[times - 2])
    return ESdata

def ES3(datalist,n,alpha):
    ESdata1=ES(datalist,n,alpha)[1:]
    ESdata2=ES(ESdata1,n,alpha)[1:]
    ESdata3=ES(ESdata2,n,alpha)[1:]
    ESpred=[]

    at =3*ESdata1[0]-3*ESdata2[0]+ESdata3[0]
    bt=alpha/(2*(1-alpha)**2)*((6-5*alpha)*ESdata1[0]-2*(5-4*alpha)*ESdata2[0]+(4-3*alpha)*ESdata3[0])
    ct=alpha**2/(2*(1-alpha)**2)*(ESdata1[0]-2*ESdata2[0]+ESdata3[0])
    # for times in range(0,min(n,len(datalist))):
    for times in range(0, n):
        ESpred.append(at+bt*times+ct*times**2)
    return ESpred

oriData = pd.read_excel('../data/' + var_name + '3_2.xlsx', sheetname='Sheet1')
oriData = oriData.fillna(0.0)
train_and_label_data = []
train_and_label_data = transform(oriData)
for predict_month,month_num in {'201609':4}.items():#{'201506':7,'201512':7,'201606':7,'201608':5}.items():
    for i in range(total_colnum,total_colnum+1):
        print 'predict_month:\t'+predict_month
        print 'month_num:\t'+str(month_num)
        # train_data = np.array(train_and_label_data)[i][:, :-1]
        # label_data = np.array(train_and_label_data)[i][:, -1]
        train_data = np.array(train_and_label_data)[i]
        alpha=0.55
        yita=0.001
        train_total = []
        for item in train_data:
            train_total.extend(item[1:])

        ###训练
        last_err=2**31-1
        for time in range(17):
            ES_total=[]
            for item in train_data:
                ES_total.extend(list(pd.ewma(item,alpha=alpha)))
            err_total=[ES_total[j]-train_total[j] for j in range(len(train_total))]
            xi=[j%6 for  j in range(len(train_total))]
            temp=[err_total[j]*xi[j] for j in range(len(train_total))]
            err_squar = [err_total[j] ** 2 for j in range(len(train_total))]
            new_err=sum(err_squar)


            alpha = alpha + yita / len(train_total) * sum(temp)
            print str(time) + '\n' + "total error:\t" + str(new_err)


            # if new_err < last_err:
            #     alpha=alpha+yita*sum(temp)
            #     last_err=new_err
            #     print str(time) + '\n' + "total error:\t" + str(new_err)
            # else:
            #     yita=yita/10
            #     time=time-1
            #     print str(new_err)+'\t'+str(last_err)+'\tbigger error'

        print alpha

        colnum = (int(predict_month) / 100 - 2014) * 12 + int(predict_month) % 100+1
        test_list = []
        test_list.extend(list(oriData.iloc[i, colnum - train_year :colnum ].values))
        # print str(ES3(test_list, train_year + month_num, alpha)[-month_num:])
        result=list(pd.ewma(np.array(test_list),alpha=alpha))
        print str(result)
        for year in range(month_num):

            # print str(int(predict_month)+year)+':\t'+str(ES(test_list, train_year+year, alpha)[-1])
            result.append(alpha*result[-1]+(1-alpha)*result[-2])
            print str(result[-1])
        # print '\n'
