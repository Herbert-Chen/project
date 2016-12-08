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
    for i in range(100):
        for j in range(1):
            print 'num:'+str(i)+'\t'+'time:'+str(j)
            test_time_series=oriData.iloc[i,:length/2+j]
            decomposition = seasonal_decompose(list(test_time_series), model="additive",
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
            # print trend
            model_trend = ARMA(trend[train_year / 2:-train_year / 2], order=(p1, q1))
            result_trend = model_trend.fit(disp=-1)
            print 'num:'+str(i)+'\t'+'time:'+str(j)+'\t'+'trend is done'
            # p2, q2 = pqorder(seasonal)
            # model_seasonal = ARMA(seasonal, order=(p2, q2))
            # result_seasonal = model_seasonal.fit(disp=-1)
            # print 'num:' + str(i) + '\t' + 'time:' + str(j) + '\t' + 'seasonal is done'
            p3, q3 = pqorder(residual[train_year / 2:-train_year / 2])
            # print residual
            model_residual = ARMA(residual[train_year / 2:-train_year / 2], order=(p3, q3))
            result_residual = model_residual.fit(disp=-1)
            print 'num:' + str(i) + '\t' + 'time:' + str(j) + '\t' + 'residual is done'

            # future_result = result_residual.predict(0, len(list(test_time_series)) - train_year / 2)[
            #                 -1] + result_seasonal.predict(
            #     0, len(list(test_time_series)) )[-1] + result_trend.predict(0,
            #                                                                 len(list(test_time_series)) - train_year / 2 )[
            #                                                   -1]
            future_result = result_residual.predict(0, len(list(test_time_series)) - train_year / 2)[
                                -1]  + result_trend.predict(0,
                                                                           len(list(
                                                                               test_time_series)) - train_year / 2)[
                                -1]-seasonal[-1]
            # future_result=result_trend.predict(0,len(list(test_time_series)) - train_year / 2 )[-1]
            error_array.append(abs(future_result-oriData.iloc[i,length/2+j])/oriData.iloc[i,length/2+j]*100)
            print error_array
            # print seasonal
