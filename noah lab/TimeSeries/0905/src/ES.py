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
from matplotlib import pyplot as plt
import random
from statsmodels.tsa.arima_model import ARMA
reload(sys)
sys.setdefaultencoding('utf-8')

from statsmodels.tsa.stattools import adfuller
import test_stationarity
xlabel = [x for x in range(33)]
# def test_stationarity(timeseries):
#     # Determing rolling statistics
#     rolmean = pd.rolling_mean(timeseries, window=6)
#     rolstd = pd.rolling_std(timeseries, window=6)
#
#     # Plot rolling statistics:
#     orig = plt.plot(xlabel,timeseries, color='blue', label='Original')
#     mean = plt.plot(xlabel,rolmean, color='red', label='Rolling Mean')
#     std = plt.plot(xlabel,rolstd, color='black', label='Rolling Std')
#     plt.legend(loc='best')
#     plt.title('Rolling Mean & Standard Deviation')
#     # plt.show(block=False)
#     plt.show()
#     # Perform Dickey-Fuller test:
#     print 'Results of Dickey-Fuller Test:'
#     dftest = adfuller(timeseries, autolag='AIC')
#     dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
#     for key, value in dftest[4].items():
#         dfoutput['Critical Value (%s)' % key] = value
#     print dfoutput


predict_month = '201506'
###两条路径，一个行数,两个序列
train_year = 12
var_name = 'A'
if var_name == 'A':
    total_colnum = 172
else:total_colnum=112

###python3里面要修改
colnum=(int(predict_month)/100-2014)*12+int(predict_month)%100

def transform(df):
    data_and_label = []
    for i in range(0, total_colnum+1):
        data_and_label.append([])
        for j in range(1, len(df.columns) - train_year):          ###参见ES_2
            flag = True
            for k in range(train_year + 1):  ##训练和标签都不包含预测月
                if str(df.columns[j + k]) == predict_month:
                    flag = False
                    # continue

            if flag == True:
                data_and_label[i].extend([list(df.iloc[i, j:(j + train_year + 1)].values)])
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


oriData = pd.read_excel('../data/' + var_name + '3_2.xlsx', sheetname='Sheet1')
oriData = oriData.fillna(0.0)

train_and_label_data = []
train_and_label_data = transform(oriData)
# for t in [x for x in range(5,100,5)]:
#     for predict_month,month_num in {'201609':4}.items():#{'201506':7,'201512':7,'201606':7,'201608':5}.items():
#         for i in range(total_colnum,total_colnum+1):
#             print 'predict_month:\t'+predict_month
#             print 'month_num:\t'+str(month_num)
#             train_data = np.array(train_and_label_data)[i][:, :-1]
#             # label_data = np.array(train_and_label_data)[i][:, -1]
#             # train_data = np.array(train_and_label_data)[i]
#             alpha=0.55
#             yita=0.001
#             train_total = []
#             for item in train_data:
#                 train_total.extend(item)
#
#             ###训练
#             last_err=2**31-1
#
#             for time in range(t):
#                 ES_total=[]
#                 for item in train_data:
#                     ES_total.extend(ES(item, train_year, alpha))
#                 err_total=[ES_total[j]-train_total[j] for j in range(len(train_total))]
#                 xi=[j%6 for  j in range(len(train_total))]
#                 temp=[err_total[j]*xi[j] for j in range(len(train_total))]
#                 err_squar = [err_total[j] ** 2 for j in range(len(train_total))]
#                 new_err=sum(err_squar)
#
#                 if new_err < last_err:
#                     if alpha-yita/len(train_total)*sum(temp)>=1:
#                         yita=yita/10
#                         time=time-1
#                     else:
#                         alpha=alpha-yita/len(train_total)*sum(temp)
#                         last_err=new_err
#                         print str(time) + '\n' + "total error:\t" + str(new_err)
#                 else:
#                     yita=yita/10
#                     time=time-1
#
#             print alpha
#
#             colnum = (int(predict_month) / 100 - 2014) * 12 + int(predict_month) % 100+1
#             test_list = []
#             test_list.extend(list(oriData.iloc[i, colnum - train_year + 1 :colnum ].values))
#             for year in range(month_num):
#
#                 # print str(int(predict_month)+year)+':\t'+str(ES(test_list, train_year+year, alpha)[-1])
#                 print str(ES(test_list, train_year + year, alpha)[-1])
#             print '\n'
i=total_colnum
# pyplot.figure()
# xlabel = [x for x in range(33)]
# test_list=list(oriData.iloc[i, 1:].values)
# rolmean = pd.rolling_mean(oriData.iloc[i, 1:], window=6)
# rolstd = pd.rolling_std(oriData.iloc[i, 1:], window=6)
#
# # temp = test_list[-8:]
# # temp.extend([test_list[-1], test_list[-1], test_list[-1], test_list[-1]])
# # pyplot.title(str(alpha))
# origin=pyplot.plot(xlabel, test_list, color='red',label='origin')
# mean=pyplot.plot(xlabel, rolmean, color='black',label='mean')
# std=pyplot.plot(xlabel, rolstd, color='green',label='std')
# # pyplot.plot(xlabel, ES(test_list, train_year + month_num, alpha)[-12:], color='black')
# # pyplot.savefig('../result/0921' + u'合并抵消' + '/train_%d_times.jpg' % t)
# pyplot.show()
# test_stationarity.draw_trend(oriData.iloc[i, 1:],6)
rol_weighted_mean = pd.ewma(oriData.iloc[i, 1:], span=6)
test_stationarity.draw_trend(oriData.iloc[i, 1:]-rol_weighted_mean,6)

# test_stationarity.testStationarity(oriData.iloc[i, 1:])
temp=np.log(oriData.iloc[i, 1:])-np.log(rol_weighted_mean)
# test_stationarity.testStationarity(oriData.iloc[i, 1:]-rol_weighted_mean)
test_stationarity.testStationarity(temp)
###查分
temp_diff=temp-temp.shift()
test_stationarity.draw_trend(temp_diff,6)
test_stationarity.testStationarity(temp_diff[1:])

###分解
from statsmodels.tsa.seasonal import seasonal_decompose
# decomposition = seasonal_decompose(oriData.iloc[i,1:], model="additive",freq=6)
decomposition = seasonal_decompose(list(oriData.iloc[i,1:].values), model="additive",freq=6)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
test_stationarity.draw_trend(pd.DataFrame(trend),6)
test_stationarity.draw_trend(pd.DataFrame(seasonal),6)
test_stationarity.draw_trend(pd.DataFrame(residual),6)
test_stationarity.testStationarity(trend[3:-3])
test_stationarity.testStationarity(seasonal)
test_stationarity.testStationarity(residual[3:-3])
temp=trend[3:-3]

test_stationarity.draw_acf_pacf(trend)
test_stationarity.draw_acf_pacf(seasonal)
temp=residual[3:-3]
test_stationarity.draw_acf_pacf(residual)

model_trend = ARMA(trend[3:-3], order=(3, 0))
result_trend = model_trend.fit(disp=-1)
pd.DataFrame(result_trend.fittedvalues).plot()
pd.DataFrame(trend[3:-3]).plot()

model_seasonal = ARMA(seasonal, order=(5,0))
result_seasonal = model_seasonal.fit(disp=-1)
pd.DataFrame(result_seasonal.fittedvalues).plot()
pd.DataFrame(seasonal).plot()

model_residual = ARMA(residual[3:-3], order=(1, 1))
result_residual = model_residual.fit(disp=-1)
pd.DataFrame(result_residual.fittedvalues).plot()
pd.DataFrame(residual[3:-3]).plot()

fit_data=result_residual.fittedvalues+result_seasonal.fittedvalues[3:-3]+result_trend.fittedvalues
plt.figure()

plt.plot(fit_data,color='red')
plt.plot(oriData.iloc[i,4:-3].values,color='black')
fit_data-oriData.iloc[i,4:-3].values

result_residual.predict(0,33)[-4:]+result_seasonal.predict(0,36)[-4:]+result_trend.predict(0,33)[-4:]