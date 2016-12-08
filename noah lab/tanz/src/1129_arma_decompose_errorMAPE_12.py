#encoding=utf-8
###依赖包


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from sklearn import cross_validation

import sys
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

import random
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.seasonal import seasonal_decompose
from collections import OrderedDict
from math import ceil
from statsmodels.tsa.arima_model import _arma_predict_out_of_sample
from statsmodels.tsa.stattools import adfuller
import test_stationarity
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
reload(sys)
sys.setdefaultencoding('utf-8')
import warnings
warnings.filterwarnings("ignore")

def median(lst):
    if not lst:
        return
    lst=sorted(lst)
    if len(lst)%2==1:
        return lst[len(lst)/2]
    else:
        return  (lst[len(lst)/2-1]+lst[len(lst)/2])/2.0

def pqorder(ts):
    # 定阶
    # 一般阶数不超过length/10
    pmax = int(len(ts)/5)
    # 一般阶数不超过length/10
    qmax = int(len(ts)/5)
    # bic矩阵
    bic_matrix = []
    for p in range(pmax):
        tmp = []
        for q in range(qmax):
            # 存在部分报错，所以用try来跳过报错。
            try:
                tmp.append(ARMA(ts, (p, q)).fit(disp=-1).bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)
    # 从中可以找出最小值
    bic_matrix = pd.DataFrame(bic_matrix)
    # 先用stack展平，然后用idxmin找出最小值位置。
    p, q = bic_matrix.stack().idxmin()
    return p,q

length=32
pred_time=16
year_list=[x for x in range(32)]
train_year=2
if __name__ == '__main__':




    ###excel读数据，train和test
    oriData = pd.read_excel('../data/data.xlsx', sheetname='Sheet1')
    oriData = oriData.fillna(0.0)

    error_array=[]
    for i in range(87,100):
        test_time_series = list(oriData.iloc[i, :length / 2])
        ext_time_series=[]
        for j in range(11):
            print 'num:'+str(i)+'\t'+'time:'+str(j)

            decomposition = seasonal_decompose(test_time_series+ext_time_series, model="additive",
                                               freq=train_year)
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid

            # # 定阶
            # # 一般阶数不超过length/10
            # pmax = int(len(diff_1) /3)
            # # 一般阶数不超过length/10
            # qmax = int(len(diff_1)/3 )
            # # bic矩阵
            # bic_matrix = []
            # for p in range(pmax):
            #     tmp = []
            #     for q in range(qmax):
            #         # 存在部分报错，所以用try来跳过报错。
            #         try:
            #             tmp.append(ARMA(diff_1, (p, q)).fit(disp=-1).bic)
            #         except:
            #             tmp.append(None)
            #     bic_matrix.append(tmp)
            # # 从中可以找出最小值
            # bic_matrix = pd.DataFrame(bic_matrix)
            # # # 先用stack展平，然后用idxmin找出最小值位置。
            # p, q = bic_matrix.stack().idxmin()
            # # print(u'BIC最小的p值和q值为：%s、%s' % (p, q))
            # model = ARMA(diff_1.T.values[0], (p, q)).fit(disp=-1)

            p1,q1=pqorder(trend[train_year / 2:-train_year / 2])
            print trend
            model_trend = ARMA(trend[train_year / 2:-train_year / 2], order=(p1, q1))
            result_trend = model_trend.fit(disp=-1)
            print 'num:'+str(i)+'\t'+'time:'+str(j)+'\t'+'trend is done'
            # p2, q2 = pqorder(seasonal)
            # model_seasonal = ARMA(seasonal, order=(p2, q2))
            # result_seasonal = model_seasonal.fit(disp=-1)
            # print 'num:' + str(i) + '\t' + 'time:' + str(j) + '\t' + 'seasonal is done'
            p3, q3 = pqorder(residual[train_year / 2:-train_year / 2])
            print residual
            model_residual = ARMA(residual[train_year / 2:-train_year / 2], order=(p3, q3))
            result_residual = model_residual.fit(disp=-1)
            print 'num:' + str(i) + '\t' + 'time:' + str(j) + '\t' + 'residual is done'

            # future_result = result_residual.predict(0, len(test_time_series+ext_time_series) - train_year / 2)[
            #                 -1] + result_seasonal.predict(
            #     0, len(test_time_series+ext_time_series))[-1] + result_trend.predict(0,
            #                                                                 len(test_time_series+ext_time_series) - train_year / 2 )[
            #                                                   -1]
            # future_result=result_trend.predict(0,len(list(test_time_series)) - train_year / 2 )[-1]
            future_result = result_residual.predict(0, len(list(test_time_series)) - train_year / 2)[
                                -1] + result_trend.predict(0,
                                                           len(list(
                                                               test_time_series)) - train_year / 2)[
                                -1] - seasonal[-1]
            ext_time_series.append(future_result)
            print 'num:' + str(i) + '\t' + 'time:' + str(j) + '\text_time_series:' + '\t' + str(ext_time_series)
        error_array.append(abs(future_result-oriData.iloc[i,length/2+11])/oriData.iloc[i,length/2+11]*100)
        # print oriData.iloc[i,length/2+11]
        print error_array
        print '\n\n'
            # # pyplot.figure()，对比预测与实际值
            # plt.figure()
            # # plt.plot(year_list[num_0-1:-1], pred[:len(test_time_series)], label='prediction', color='red')
            # # plt.plot(year_list[num_0-1:-1], test_time_series, label='ground truth', color='black')
            # plt.plot(year_list[length/2:], pred[:len(test_time_series)], label='prediction', color='red')
            # plt.plot(year_list[length/2:], test_time_series, label='ground truth', color='black')
            # # plt.plot( pred, label='prediction', color='red')
            # # plt.plot( test_time_series, label='ground truth', color='black')
            # plt.title('time_series_' + str(i) + '_' +  '_contrast.png')
            # plt.xlabel('year')
            # plt.savefig('../result/1126/time_series_' + str(i) + '_' +  '_contrast.png')
            # plt.legend()
            # plt.close()


    # print median(error_array)
    # print len(error_array)
    # print error_array
        # ###预测16-20
        # future_Data = pd.read_excel('../data/future_'+var_name+'.xlsx', sheetname='Sheet1')
        # pred_future = model.predict(np.array(future_Data.values))
        # print 'industry_label:%s'%str(industry_labels)
        # print pred_future