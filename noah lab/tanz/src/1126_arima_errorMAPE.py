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
length=32
pred_time=16
year_list=[x for x in range(32)]
if __name__ == '__main__':




    ###excel读数据，train和test
    oriData = pd.read_excel('../data/data.xlsx', sheetname='Sheet1')
    oriData = oriData.fillna(0.0)

    error_array=[]
    for i in range(87,88):
        for j in range(pred_time):
            print 'num:'+str(i)+'\t'+'time:'+str(j)
            test_time_series=oriData.iloc[i,:length/2+j]
            # temp=list(test_time_series)

            # test_time_series=np.array(temp)
            # test_stationarity.testStationarity(test_time_series)
            # test_stationarity.draw_ts(test_time_series)
            # pd.DataFrame(test_time_series).plot()
            diff = pd.DataFrame(test_time_series).diff().dropna()
            # diff.plot()
            diff_1 = pd.DataFrame(diff).diff().dropna()
            # diff_1.plot()
            # diff_2 = pd.DataFrame(test_time_series).diff(2).dropna()
            # diff_2.columns = [var_name]
            # diff_2.plot()
            # test_stationarity.testStationarity(diff_1.T.values[0])
            # print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(diff_1, lags=1))

            # 定阶
            # 一般阶数不超过length/10
            pmax = int(len(diff_1) /3)
            # 一般阶数不超过length/10
            qmax = int(len(diff_1)/3 )
            # bic矩阵
            bic_matrix = []
            for p in range(pmax):
                tmp = []
                for q in range(qmax):
                    # 存在部分报错，所以用try来跳过报错。
                    try:
                        tmp.append(ARMA(diff_1.T.values[0], (p, q)).fit(disp=-1).bic)
                    except:
                        tmp.append(None)
                bic_matrix.append(tmp)
            # 从中可以找出最小值
            bic_matrix = pd.DataFrame(bic_matrix)
            # 先用stack展平，然后用idxmin找出最小值位置。
            p, q = bic_matrix.stack().idxmin()
            # print(u'BIC最小的p值和q值为：%s、%s' % (p, q))
            model = ARMA(diff_1.T.values[0], (p, q)).fit(disp=-1)
            # 给出一份模型报告
            # model.summary2()
            # 作为期5天的预测，返回预测结果、标准误差、置信区间。
            # model.forecast()
            pred_2=list(model.predict(0,model.predict(0).shape[0]+1-1))
            pred_1=list(diff.values[0])
            for num in range(len(diff.values)):
                # print str(pred_2[num])+'\t'+str(float(diff.values[num]))+'\t'+str(pred_2[num]+float(diff.values[num]))
                pred_1.append(pred_2[num]+float(diff.values[num]))
            # for num in range(1):
            #     pred_1.append(pred_2[num+model.predict(0).shape[0]] + pred_1[num+model.predict(0).shape[0]])

            pred=[test_time_series.iloc[0]]
            for num in range(len(test_time_series)):
                pred.append(pred_1[num]+float(test_time_series.iloc[num]))
            # for num in range(1):
            #     pred.append(pred_1[num+len(test_time_series)] + pred[num+len(test_time_series)])


            # test=test_stationarity.testStationarity(diff_1.T.values[0])
            # print '平稳检验p值：'+str(test['p-value'])
            # print(u'BIC最小的p值和q值为：%s、%s' % (p, q))
            # print 'data:pred-real'
            squ_error = 0
            # for num in range(len(test_time_series)):
            #     print str(year_list[num])+'\t'+str(pred[num])+'\t'+str(test_time_series.iloc[num])


            # print '误差绝对值和：'+str(squ_error)
            # print '未来5年：'
            # for num in range(1):
            #     print pred[len(test_time_series)+num]

            error_array.append(abs(pred[-1]-oriData.iloc[i,length/2+j])/oriData.iloc[i,length/2+j]*100)
            print error_array

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