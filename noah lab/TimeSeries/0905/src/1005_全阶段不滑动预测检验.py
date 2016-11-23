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
from statsmodels.tsa.seasonal import seasonal_decompose
from collections import OrderedDict
from math import ceil
from statsmodels.tsa.arima_model import _arma_predict_out_of_sample



# import warnings
# warnings.filterwarnings("ignore")



reload(sys)
sys.setdefaultencoding('utf-8')

from statsmodels.tsa.stattools import adfuller
import test_stationarity
xlabel = [x for x in range(33)]



predict_month = '201611'
###两条路径，一个行数,两个序列
train_year = 12
var_name = 'A'
if var_name == 'A':
    total_colnum = 172
    total_month=33
else:
    total_colnum=112
    total_month=33

###python3里面要修改，pd里面当前列号-1，如果要计数需要
colnum=(int(predict_month)/100-2014)*12+int(predict_month)%100+1
last_colnum=(int(201610)/100-2014)*12+int(201610)%100+1
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



oriData = pd.read_excel('../data/' + var_name + '3_2.xlsx', sheetname='Sheet1')
oriData = oriData.fillna(0.0)

train_and_label_data = []
train_and_label_data = transform(oriData)

# for predict_month,month_num in {'201512':7}.items():#{'201506':7,'201512':7,'201606':7,'201608':5}.items():
#     for i in range(total_colnum,total_colnum+1):
#         rol_weighted_mean = pd.ewma(oriData.iloc[i, 1:], span=6)
#         # rol_weighted_mean = oriData.iloc[i, 1:].ewm(span=6).mean()
#         # test_stationarity.draw_trend(oriData.iloc[i, 1:]-rol_weighted_mean,train_year)
#
#
#         ###分解
#         decomposition = seasonal_decompose(list(oriData.iloc[i,1:].values), model="additive",freq= 6)
#         trend = decomposition.trend
#         seasonal = decomposition.seasonal
#         residual = decomposition.resid
#         # test_stationarity.draw_trend(pd.DataFrame(trend),train_year)
#         # test_stationarity.draw_trend(pd.DataFrame(seasonal),train_year)
#         # test_stationarity.draw_trend(pd.DataFrame(residual),train_year)
#         # test_stationarity.testStationarity(trend[train_year/2:-train_year/2])
#         # test_stationarity.testStationarity(seasonal)
#         # test_stationarity.testStationarity(residual[train_year/2:-train_year/2])
#
#         #acf,pacf作图
#         # test_stationarity.draw_acf_pacf(trend)
#         # test_stationarity.draw_acf_pacf(seasonal)
#         # test_stationarity.draw_acf_pacf(residual)
#
#         model_trend = ARMA(trend[train_year/2:-train_year/2], order=(1, 0))
#         result_trend = model_trend.fit(disp=-1)
#         # pd.DataFrame(result_trend.fittedvalues).plot()
#         # pd.DataFrame(trend[train_year/2:-train_year/2]).plot()
#
#         model_seasonal = ARMA(seasonal, order=(4,0))
#         result_seasonal = model_seasonal.fit(disp=-1)
#         # pd.DataFrame(result_seasonal.fittedvalues).plot()
#         # pd.DataFrame(seasonal).plot()
#
#         model_residual = ARMA(residual[train_year/2:-train_year/2], order=(1, 0))
#         result_residual = model_residual.fit(disp=-1)
#         # pd.DataFrame(result_residual.fittedvalues).plot()
#         # pd.DataFrame(residual[train_year/2:-train_year/2]).plot()
#
#         fit_data=result_residual.fittedvalues+result_seasonal.fittedvalues[train_year/2:-train_year/2]+result_trend.fittedvalues
#         plt.figure()
#         plt.title(var_name+'_'+str(train_year)+'_contrast')
#         plt.plot(fit_data,color='red')
#         plt.plot(oriData.iloc[i,train_year/2+1:-train_year/2].values,color='black')
#         plt.savefig('../result/0927'+u'合并抵消'+'/'+var_name+'_'+str(train_year)+'_contrast.png')
#         for j in range(total_month-train_year):
#             print str(oriData.columns[1+train_year/2+j])+':\t'+str(fit_data[j])+'\t'+str(oriData.iloc[i,1+train_year/2+j])
#         print
#         print 'the next %d month:'%month_num
#         future_result=result_residual.predict(0, total_month - train_year / 2 + month_num - 1)[-month_num:] + result_seasonal.predict(
#             0, total_month + month_num - 1)[-month_num:] + result_trend.predict(0,
#                                                                                 total_month - train_year / 2 + month_num - 1)[
#                                                            -month_num:]
#         for k in range(month_num):
#             print future_result[k]
#

###验证初始条件下偏差对后续的影响
# for predict_month,month_num in {'201512':7}.items():#{'201506':7,'201512':7,'201606':7,'201608':5}.items():
#     for i in range(total_colnum,total_colnum+1):
#         print predict_month
#         print month_num
#         print
#         colnum = (int(predict_month) / 100 - 2014) * 12 + int(predict_month) % 100 + 1
#         rol_weighted_mean = pd.ewma(oriData.iloc[i, 1:colnum], span=6)
#         # rol_weighted_mean = oriData.iloc[i, 1:].ewm(span=6).mean()
#         # test_stationarity.draw_trend(oriData.iloc[i, 1:]-rol_weighted_mean,train_year)
#
#
#         ###分解
#         decomposition = seasonal_decompose(list(oriData.iloc[i,1:colnum].values), model="additive",freq= train_year)
#         trend = decomposition.trend
#         seasonal = decomposition.seasonal
#         residual = decomposition.resid
#         # test_stationarity.draw_trend(pd.DataFrame(trend),train_year)
#         # test_stationarity.draw_trend(pd.DataFrame(seasonal),train_year)
#         # test_stationarity.draw_trend(pd.DataFrame(residual),train_year)
#         # test_stationarity.testStationarity(trend[train_year/2:-train_year/2])
#         # test_stationarity.testStationarity(seasonal)
#         # test_stationarity.testStationarity(residual[train_year/2:-train_year/2])
#
#         #acf,pacf作图
#         # test_stationarity.draw_acf_pacf(trend)
#         # test_stationarity.draw_acf_pacf(seasonal)
#         # test_stationarity.draw_acf_pacf(residual)
#
#         model_trend = ARMA(trend[train_year/2:-train_year/2], order=(1, 0))
#         result_trend = model_trend.fit(disp=-1)
#         # pd.DataFrame(result_trend.fittedvalues).plot()
#         # pd.DataFrame(trend[train_year/2:-train_year/2]).plot()
#
#         model_seasonal = ARMA(seasonal, order=(5,0))
#         result_seasonal = model_seasonal.fit()
#         # pd.DataFrame(result_seasonal.fittedvalues).plot()
#         # pd.DataFrame(seasonal).plot()
#
#         model_residual = ARMA(residual[train_year/2:-train_year/2], order=(1, 0))
#         result_residual = model_residual.fit(disp=-1)
#         # pd.DataFrame(result_residual.fittedvalues).plot()
#         # pd.DataFrame(residual[train_year/2:-train_year/2]).plot()
#
#         # fit_data=result_residual.fittedvalues+result_seasonal.fittedvalues[train_year/2:-train_year/2]+result_trend.fittedvalues
#         # plt.figure()
#         # plt.title(var_name+'_'+str(train_year)+'_'+predict_month+str(month_num)+'_contrast')
#         # plt.plot(fit_data,color='red')
#         # plt.plot(oriData.iloc[i,train_year/2+1:-train_year/2].values,color='black')
#         # plt.savefig('../result/0927'+u'合并抵消'+'/'+var_name+'_'+str(train_year)+'_'+predict_month+str(month_num)+'_contrast.png')
#         # for j in range(total_month-train_year):
#         #     print str(oriData.columns[1+train_year/2+j])+':\t'+str(fit_data[j])+'\t'+str(oriData.iloc[i,1+train_year/2+j])
#         # print
#         # print 'the next %d month:'%month_num
#
#         future_result=result_residual.predict(0, colnum-1 - train_year / 2 + month_num - 1)[-month_num:] + result_seasonal.predict(
#             0, colnum -1+ month_num - 1)[-month_num:] + result_trend.predict(0,
#                                                                                 colnum-1 - train_year / 2 + month_num - 1)[
#                                                            -month_num:]
#
#         params = result_trend.params
#         r = result_trend.resid
#         p = result_trend.k_ar
#         q = result_trend.k_ma
#         k_exog = result_trend.k_exog
#         k_trend = result_trend.k_trend
#         steps = month_num
#         y=result_trend.predict(0,colnum-1 - train_year / 2 - 1)
#         y[-1]=y[-1]-5
#         trend_pred=_arma_predict_out_of_sample(params, steps, r, p, q, k_trend, k_exog, endog=y, exog=None, start=len(y))
#
#         params = result_residual.params
#         r = result_residual.resid
#         p = result_residual.k_ar
#         q = result_residual.k_ma
#         k_exog = result_residual.k_exog
#         k_trend = result_residual.k_trend
#         steps = month_num
#         y = result_trend.predict(0, colnum - 1 - train_year / 2 - 1)
#         y[-1] = y[-1] - 5
#         trend_pred = _arma_predict_out_of_sample(params, steps, r, p, q, k_trend, k_exog, endog=y, exog=None,
#                                                  start=len(y))
#
#         params = result_trend.params
#         r = result_trend.resid
#         p = result_trend.k_ar
#         q = result_trend.k_ma
#         k_exog = result_trend.k_exog
#         k_trend = result_trend.k_trend
#         steps = month_num
#         y = result_trend.predict(0, colnum - 1 - train_year / 2 - 1)
#         y[-1] = y[-1] - 5
#         trend_pred = _arma_predict_out_of_sample(params, steps, r, p, q, k_trend, k_exog, endog=y, exog=None,
#                                                  start=len(y))
#         # for k in range(month_num):
#         #     print future_result[k]
#         # colnum = (int(predict_month) / 100 - 2014) * 12 + int(predict_month) % 100 + 1
#         plt.figure()
#         plt.title(var_name + '_' + str(train_year) + '_' + predict_month +'_'+ str(month_num) + '_contrast')
#         plt.plot(future_result, color='red')
#         plt.plot(oriData.iloc[i, colnum:colnum+month_num].values, color='black')
#         plt.savefig('../result/0927' + u'合并抵消' + '/' + var_name + '_' + str(train_year) + '_' + predict_month +'_'+ str(
#             month_num) + '_contrast.png')
#         for j in range(month_num):
#             print str(oriData.columns[colnum + j]) + ':\t' + str(future_result[j]) + '\t' + str(
#                 oriData.iloc[i,colnum + j])
#         print
#         # print 'the next %d month:' % month_num

# for predict_month,month_num in {'201512':4}.items():#{'201506':7,'201512':7,'201606':7,'201608':5}.items():
#     for i in range(total_colnum,total_colnum+1):
#         print predict_month
#         print month_num
#         print
#         colnum = (int(predict_month) / 100 - 2014) * 12 + int(predict_month) % 100 + 1
#         rol_weighted_mean = pd.ewma(oriData.iloc[i, 1:colnum], span=6)
#         # rol_weighted_mean = oriData.iloc[i, 1:].ewm(span=6).mean()
#         # test_stationarity.draw_trend(oriData.iloc[i, 1:]-rol_weighted_mean,train_year)
#
#
#         ###分解
#         decomposition = seasonal_decompose(list(oriData.iloc[i,1:colnum].values), model="additive",freq= 6)
#         trend = decomposition.trend
#         seasonal = decomposition.seasonal
#         residual = decomposition.resid
#         # test_stationarity.draw_trend(pd.DataFrame(trend),train_year)
#         # test_stationarity.draw_trend(pd.DataFrame(seasonal),train_year)
#         # test_stationarity.draw_trend(pd.DataFrame(residual),train_year)
#         # test_stationarity.testStationarity(trend[train_year/2:-train_year/2])
#         # test_stationarity.testStationarity(seasonal)
#         # test_stationarity.testStationarity(residual[train_year/2:-train_year/2])
#
#         #acf,pacf作图
#         # test_stationarity.draw_acf_pacf(trend)
#         # test_stationarity.draw_acf_pacf(seasonal)
#         # test_stationarity.draw_acf_pacf(residual)
#
#         model_trend = ARMA(trend[train_year/2:-train_year/2], order=(1, 0))
#         result_trend = model_trend.fit(disp=-1)
#         pd.DataFrame(result_trend.fittedvalues).plot()
#         pd.DataFrame(trend[train_year/2:-train_year/2]).plot()
#
#         model_seasonal = ARMA(seasonal, order=(4,0))
#         result_seasonal = model_seasonal.fit(disp=-1)
#         pd.DataFrame(result_seasonal.fittedvalues).plot()
#         pd.DataFrame(seasonal).plot()
#
#         model_residual = ARMA(residual[train_year/2:-train_year/2], order=(1, 0))
#         result_residual = model_residual.fit(disp=-1)
#         pd.DataFrame(result_residual.fittedvalues).plot()
#         pd.DataFrame(residual[train_year/2:-train_year/2]).plot()
#
#         # fit_data=result_residual.fittedvalues+result_seasonal.fittedvalues[train_year/2:-train_year/2]+result_trend.fittedvalues
#         # plt.figure()
#         # plt.title(var_name+'_'+str(train_year)+'_'+predict_month+str(month_num)+'_contrast')
#         # plt.plot(fit_data,color='red')
#         # plt.plot(oriData.iloc[i,train_year/2+1:-train_year/2].values,color='black')
#         # plt.savefig('../result/0927'+u'合并抵消'+'/'+var_name+'_'+str(train_year)+'_'+predict_month+str(month_num)+'_contrast.png')
#         # for j in range(total_month-train_year):
#         #     print str(oriData.columns[1+train_year/2+j])+':\t'+str(fit_data[j])+'\t'+str(oriData.iloc[i,1+train_year/2+j])
#         # print
#         # print 'the next %d month:'%month_num
#
#         future_result=result_residual.predict(0, colnum-1 - train_year / 2 + month_num - 1)[-month_num:] + result_seasonal.predict(
#             0, colnum -1+ month_num - 1)[-month_num:] + result_trend.predict(0,
#                                                                                 colnum-1 - train_year / 2 + month_num - 1)[
#                                                            -month_num:]
#         # for k in range(month_num):
#         #     print future_result[k]
#         # colnum = (int(predict_month) / 100 - 2014) * 12 + int(predict_month) % 100 + 1
#         # plt.figure()
#         # plt.title(var_name + '_' + str(train_year) + '_' + predict_month +'_'+ str(month_num) + '_contrast')
#         # plt.plot(future_result, color='red')
#         # plt.plot(oriData.iloc[i, colnum:colnum+month_num].values, color='black')
#         # plt.savefig('../result/0927' + u'合并抵消' + '/' + var_name + '_' + str(train_year) + '_' + predict_month +'_'+ str(
#         #     month_num) + '_contrast.png')
#         for j in range(month_num):
#             print str(oriData.columns[colnum + j]) + ':\t' + str(future_result[j]) + '\t' + str(
#                 oriData.iloc[i,colnum + j])
#         print
#         print 'the next %d month:' % month_num
#         for k in range(month_num):
#             print future_result[k]



for predict_month,month_num in {'201611':2}.items():#{'201506':7,'201512':7,'201606':7,'201608':5}.items():
    for i in range(total_colnum,total_colnum+1):
        print predict_month
        print month_num
        print
        colnum = (int(predict_month) / 100 - 2014) * 12 + int(predict_month) % 100 + 1
        rol_weighted_mean = pd.ewma(oriData.iloc[i, 1:colnum], span=6)
        # rol_weighted_mean = oriData.iloc[i, 1:].ewm(span=6).mean()
        # test_stationarity.draw_trend(oriData.iloc[i, 1:]-rol_weighted_mean,train_year)


        ###分解
        decomposition = seasonal_decompose(list(oriData.iloc[i,1:colnum].values), model="additive",freq= train_year)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        # test_stationarity.draw_trend(pd.DataFrame(trend),train_year)
        # test_stationarity.draw_trend(pd.DataFrame(seasonal),train_year)
        # test_stationarity.draw_trend(pd.DataFrame(residual),train_year)
        # test_stationarity.testStationarity(trend[train_year/2:-train_year/2])
        # test_stationarity.testStationarity(seasonal)
        # test_stationarity.testStationarity(residual[train_year/2:-train_year/2])

        #acf,pacf作图
        # test_stationarity.draw_acf_pacf(trend)
        # test_stationarity.draw_acf_pacf(seasonal)
        # test_stationarity.draw_acf_pacf(residual)

        model_trend = ARMA(trend[train_year/2:-train_year/2], order=(1, 0))
        result_trend = model_trend.fit(disp=-1)
        pd.DataFrame(result_trend.fittedvalues).plot()
        pd.DataFrame(trend[train_year/2:-train_year/2]).plot()

        model_seasonal = ARMA(seasonal, order=(4,0))
        result_seasonal = model_seasonal.fit()
        pd.DataFrame(result_seasonal.fittedvalues).plot()
        pd.DataFrame(seasonal).plot()

        model_residual = ARMA(residual[train_year/2:-train_year/2], order=(1, 0))
        result_residual = model_residual.fit(disp=-1)
        pd.DataFrame(result_residual.fittedvalues).plot()
        pd.DataFrame(residual[train_year/2:-train_year/2]).plot()

        # fit_data=result_residual.fittedvalues+result_seasonal.fittedvalues[train_year/2:-train_year/2]+result_trend.fittedvalues
        # plt.figure()
        # plt.title(var_name+'_'+str(train_year)+'_'+predict_month+str(month_num)+'_contrast')
        # plt.plot(fit_data,color='red')
        # plt.plot(oriData.iloc[i,train_year/2+1:-train_year/2].values,color='black')
        # plt.savefig('../result/0927'+u'合并抵消'+'/'+var_name+'_'+str(train_year)+'_'+predict_month+str(month_num)+'_contrast.png')
        # for j in range(total_month-train_year):
        #     print str(oriData.columns[1+train_year/2+j])+':\t'+str(fit_data[j])+'\t'+str(oriData.iloc[i,1+train_year/2+j])
        # print
        # print 'the next %d month:'%month_num

        future_result=result_residual.predict(0, colnum-1 - train_year / 2 + month_num - 1)[-month_num:] + result_seasonal.predict(
            0, colnum -1+ month_num - 1)[-month_num:] + result_trend.predict(0,
                                                                                colnum-1 - train_year / 2 + month_num - 1)[
                                                           -month_num:]
        # for k in range(month_num):
        #     print future_result[k]
        # colnum = (int(predict_month) / 100 - 2014) * 12 + int(predict_month) % 100 + 1
        plt.figure()
        plt.title(var_name + '_' + str(train_year) + '_' + predict_month +'_'+ str(month_num) + '_contrast')
        plt.plot(future_result, color='red')
        plt.plot(oriData.iloc[i, colnum:colnum+month_num].values, color='black')
        plt.savefig('../result/0927' + u'合并抵消' + '/' + var_name + '_' + str(train_year) + '_' + predict_month +'_'+ str(
            month_num) + '_contrast.png')
        # for j in range(month_num):
        #     print str(oriData.columns[colnum + j]) + ':\t' + str(future_result[j]) + '\t' + str(
        #         oriData.iloc[i,colnum + j])
        print
        for j in range(month_num):
            print str(oriData.columns[colnum + j]) + ':\t' + str(future_result[j])
        # print 'the next %d month:' % month_num




# print 'train year:'+str(train_year)
#
# month_num=2
# for i in range(total_colnum,total_colnum+1):
#
#     data_set=oriData.iloc[i,1:36].values
#
#
#         ###分解
#     decomposition = seasonal_decompose(list(data_set), model="additive",filt=None,freq=int(train_year))
#     trend = decomposition.trend
#     seasonal = decomposition.seasonal
#     residual = decomposition.resid
#     # test_stationarity.draw_trend(pd.DataFrame(trend),train_year)
#     # test_stationarity.draw_trend(pd.DataFrame(seasonal),train_year)
#     # test_stationarity.draw_trend(pd.DataFrame(residual),train_year)
#     # test_stationarity.testStationarity(trend[train_year/4:-train_year/4])
#     # test_stationarity.testStationarity(seasonal)
#     # test_stationarity.testStationarity(residual[train_year/4:-train_year/4])
#
#     #acf,pacf作图
#     # test_stationarity.draw_acf_pacf(trend)
#     # test_stationarity.draw_acf_pacf(seasonal)
#     # test_stationarity.draw_acf_pacf(residual)
#
#     model_trend = ARMA(trend[train_year/2:-train_year/2], order=(1, 0))
#     result_trend = model_trend.fit(disp=-1)
#     # pd.DataFrame(result_trend.fittedvalues).plot()
#     # pd.DataFrame(trend[train_year/4:-train_year/4]).plot()
#
#     model_seasonal = ARMA(seasonal, order=(4,0))
#     result_seasonal = model_seasonal.fit(disp=-1)
#     # pd.DataFrame(result_seasonal.fittedvalues).plot()
#     # pd.DataFrame(seasonal).plot()
#
#     # residual=[residual[x]+random.random()/1000 for x in range(len(residual))]
#     model_residual = ARMA(residual[train_year/2:-train_year/2], order=(1, 1))
#     result_residual = model_residual.fit(disp=-1)
#     # pd.DataFrame(result_residual.fittedvalues).plot()
#     # pd.DataFrame(residual[train_year/4:-train_year/4]).plot()
#
#     # fit_data=result_residual.fittedvalues+result_seasonal.fittedvalues[train_year/4:-train_year/4]+result_trend.fittedvalues
#     # plt.figure()
#     # plt.title(var_name+'_'+str(train_year)+'_'+predict_month+str(month_num)+'_contrast')
#     # plt.plot(fit_data,color='red')
#     # plt.plot(oriData.iloc[i,train_year/4+1:-train_year/4].values,color='black')
#     # plt.savefig('../result/0927'+u'合并抵消'+'/'+var_name+'_'+str(train_year)+'_'+predict_month+str(month_num)+'_contrast.png')
#     # for j in range(total_month-train_year):
#     #     print str(oriData.columns[1+train_year/4+j])+':\t'+str(fit_data[j])+'\t'+str(oriData.iloc[i,1+train_year/4+j])
#     # print
#     # print 'the next %d month:'%month_num
#
#     future_result=result_residual.predict(0, colnum-1 - train_year / 2 + month_num - 1)[-month_num:] + result_seasonal.predict(
#         0, colnum -1+ month_num - 1)[-month_num:] + result_trend.predict(0,
#                                                                             colnum-1 - train_year / 2 + month_num - 1)[
#                                                        -month_num:]
#     for k in range(month_num):
#         print future_result[k]
#     # colnum = (int(predict_month) / 100 - 2014) * 12 + int(predict_month) % 100 + 1
#
#
#
#
# # plt.figure()
# # plt.title(var_name + '_' + str(train_year) + '_' + slide_pred['year_start']+'_'+slide_pred['year_end'] +'_size_'+str(slide_pred['size'])  + '_contrast')
# # plt.plot(pred_data[:length], color='red')
# # plt.plot(real_data[:length], color='black')
# # plt.savefig('../result/1017' + u'合并抵消' + '/' + var_name + '_' + str(train_year) + '_' + slide_pred['year_start']+'_'+slide_pred['year_end'] +'_size_'+str(slide_pred['size'])  + '_contrast.png')
# # print length
# # for j in range(length):
# #     print str(month_list[j]) + ':\t' + str(pred_data[j]) + '\t' + str(real_data[j])+'\t'+str(pred_data[j]-real_data[j])+'\t'+str((pred_data[j]-real_data[j])/real_data[j]*100)+'%'
# # print
# #
# # for x in range(length,len(pred_data)):
# #     print pred_data[x]
#
#
