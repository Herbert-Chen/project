#encoding=utf-8
###依赖包


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from sklearn import cross_validation
from plot import ArkPlot
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

algo = 'gbrt'   # rf, gbrt


###路径，使用前一定要检查文件夹是否存在不然报错
var_name='Y7'         ###变量名，Y1，Y2...
folder_name='../result/0827/'+var_name+'/'      ###文件夹名,最后一定要包含/

###文件中定义的格式：
#   =  folder_name + 'cv'+'_'+ var_name + '_' + label'+str(industry_label) + '_' + algo + '_start_' + str(year_start) + '_end_' + str(year_end) + '.png'

###标签,为list，判断是否存在其中，以后如果有多个标签可以扩展为dict
# industry_labels=2

###生成路径，function为 cv,contrast,test...,type为png，txt
def gen_path(function,type):
    return folder_name + function +'_'+ var_name  + '_label_'+ str(industry_labels) + '_' + algo + '_from_' + str(year_start) + '_to_' + str(year_end) + '.'+type



###y变量，所有变量列号表示，均为excel表中查到的列号，不做任何处理，随时间加减，list中减1均在后面处理,          主函数中
# col_name_dict={'Total Cash & ST Investments':456,'Total Cash & ST Investments/Total Revenue %':576,'Total Cash & ST Investments/Avg. monthly cash outflow Month':676, \
#                    'Net Cash 1 = Total Cash & ST Investments - Debt':716,'Total Cash & ST Investments/Total Assets %':736 ,'Net Cash Burn Ratio 1=Net Cash 1/ Avg. monthly cash outflow %':756, \
#                    'Net Cash 1 / Sales %':776}
# col_name_dict={'Total Cash & ST Investments/Total Revenue %':576}
###x特征，分为单一变量和随时间变量，规则同上，处理直接在下面
###此为全部变量
# otherFeatureList = [ 22, 23, 24, 25,27]                 ###删掉12,13,12在后面有，13为标签
# baseFeatureList = [36, 56, 76, 96, 116, 136, 156, 176, 196, 216, 236, 256, 276,
#                        296, 316, 336, 356, 376, 396, 416, 436, 476, 496, 516, 536,
#                        556, 596, 616, 636, 656, 796, 816, 836, 856, 876, 896, 916]
# if var_name=='Y2':
# #削减变量Y2
#     col_name_dict={'Total Cash & ST Investments/Total Revenue %':576}
#     otherFeatureList = [23, 25]
#     baseFeatureList = [56, 136, 156, 256, 276, 296, 336, 416, 436, 496, 516, 556, 856]
# elif var_name=='Y3':
# #削减变量Y3
#     col_name_dict={'Total Cash & ST Investments/Avg. monthly cash outflow Month':676}
#     otherFeatureList = [23, 25]
#     baseFeatureList = [56, 116, 136, 156, 256, 276, 296, 336, 416, 436, 496, 516, 556, 856]
# elif var_name=='Y5':
# #削减变量Y5
#     col_name_dict={'Total Cash & ST Investments/Total Assets %':736}
#     otherFeatureList = [22,23, 25]
#     baseFeatureList = [56, 136, 156, 256, 276, 296, 336, 416, 496, 516, 556, 896]
#
# elif var_name=='Y6':
# # #削减变量Y6
#     col_name_dict={'Net Cash Burn Ratio 1=Net Cash 1/ Avg. monthly cash outflow %':756}
#     otherFeatureList = [23, 25]
#     baseFeatureList = [56, 76, 136, 156, 256, 276, 296, 336, 416, 436, 496, 516, 556, 896]
# elif var_name=='Y7':
# #削减变量Y7
#     col_name_dict={'Net Cash 1 / Sales %':776}
#     otherFeatureList = [23, 25]
#     baseFeatureList = [56, 76, 116, 136, 156, 256, 276, 296, 416, 436, 496, 516, 556, 896]
#





#####基本函数

#######
# Function: Generate column numbers of each subset (by year)
# Input: labelCol: int, baseFeatureList: list[int], otherFeatureList: list[int]
# Output: featureLabelCols: list[list[int]]
#######
def gen_feature_column_numbers(labelCol, baseFeatureList, otherFeatureList):
    featureLabelCols = []
    for step in xrange(year_last):                  ###
        featureLabelColsByYear = []
        labelColNum = labelCol + step
        featureColNum = [i + step for i in baseFeatureList]
        featureLabelColsByYear.extend(otherFeatureList)
        featureLabelColsByYear.extend(featureColNum)
        featureLabelColsByYear.extend([labelColNum])
        featureLabelCols.append(featureLabelColsByYear)

    return featureLabelCols

#######
# Function: merge sub datasets into one
# Input: df: pandas.DataFrame, colsList: list[list[int]]
# Output: upperArray: np.array, col_names: list[string]
#######
def get_cleaned_dataset(df_train, df_test,colsList, industry_label):
    if industry_label !=[]:
        # df_train = df_train[ df_train['Industry Dummy'] in industry_label]
        df_train = df_train[df_train['Industry Dummy'] == industry_label]       ###扩展！！！！！
    else:
        pass
    col_names = df_train.columns[colsList[0]]
    count = 0
    # if df_test == 0:
    #     for indexes in colsList:
    #         tmpArray = np.array(df_train[indexes].values)
    #         if count == 0:
    #             upperArray = tmpArray
    #             count += 1
    #             continue
    #         else:
    #             upperArray = np.vstack((upperArray, tmpArray))
    #             count += 1
    #
    #     return upperArray, col_names
    # else:
    for indexes in colsList:
        tmpArray = np.array(df_train.iloc[:,indexes].values)
        testTmpArray = np.array(df_test[indexes].values)
        if count == 0:
            upperArray = tmpArray
            upperTestArray = testTmpArray
            count += 1
            continue
        else:
            upperArray = np.vstack((upperArray, tmpArray))
            upperTestArray = np.vstack((upperTestArray, testTmpArray))
            count += 1

    return upperArray, upperTestArray, col_names


#######
# Function: split features and label from dataset
# Input: data_set: np.array
# Output: features: np.array, label: np.array
#######
def split_feature_label(data_set):
    features = data_set[:, :-1]
    label = data_set[:, -1]
    return features, label

def gen_pearson_column_numbers(labelCol, baseFeatureList):
    colNeedIndexes = []
    for index in baseFeatureList:
        colNeedIndexes.extend([index + i for i in xrange(year_last)])
    colNeedIndexes.extend([labelCol + j for j in xrange(year_last)])
    return colNeedIndexes

#######
# Function: compute abs error between predictions and ground truths
# Input: true_y: np.array, pred_y: np.array
# Output: error: np.array
#######
def compute_error(true_y, pred_y):
    error = []
    for i in xrange(true_y.shape[0]):
        error.append(abs(true_y[i] - pred_y[i]))

    return np.array(error)


if __name__ == '__main__':
    pred_year=5
    industry_labels=[]
    for year_start in [1997]:
        if var_name == 'Y2':
            # 削减变量Y2
            col_name_dict = {'Total Cash & ST Investments/Total Revenue %': 576}
            otherFeatureList = [12, 25]
            baseFeatureList = [56, 136, 156, 256, 276, 296, 336, 416, 436, 496, 516, 556, 896]
        elif var_name == 'Y3':
            # 削减变量Y3
            col_name_dict = {'Total Cash & ST Investments/Avg. monthly cash outflow Month': 676}
            otherFeatureList = [12, 25]
            baseFeatureList = [56, 116, 136, 156, 256, 276, 296, 336, 416, 436, 496, 516, 556, 896]
        elif var_name == 'Y5':
            # 削减变量Y5
            col_name_dict = {'Total Cash & ST Investments/Total Assets %': 736}
            otherFeatureList = [22, 12, 25]
            baseFeatureList = [56, 136, 156, 256, 276, 296, 336, 416, 496, 516, 556, 896]
        elif var_name == 'Y6':
            # #削减变量Y6
            col_name_dict = {'Net Cash Burn Ratio 1=Net Cash 1/ Avg. monthly cash outflow %': 756}
            otherFeatureList = [12, 25]
            baseFeatureList = [56, 76, 136, 156, 256, 276, 296, 336, 416, 436, 496, 516, 556, 896]
        elif var_name == 'Y7':
            # 削减变量Y7
            col_name_dict = {'Net Cash 1 / Sales %': 776}
            otherFeatureList = [12, 25]
            baseFeatureList = [56, 76, 116, 136, 156, 256, 276, 296, 416, 436, 496, 516, 556, 896]

            ###x标签变换
        elif var_name == 'Y1':
            # 削减变量Y7
            col_name_dict = {'Total Cash & ST Investments':456}
            otherFeatureList = [12, 25]
            baseFeatureList = [56, 76, 116, 136, 156, 256, 276, 296, 416, 436, 496, 516, 556, 896]

            ###x标签变换
        elif var_name == 'Y4':
            # 削减变量Y7
            col_name_dict = {'Net Cash 1 = Total Cash & ST Investments - Debt':716}
            otherFeatureList = [12, 25]
            baseFeatureList = [56, 76, 116, 136, 156, 256, 276, 296, 416, 436, 496, 516, 556, 896]
            ###x标签变换
        year_end = 2016
        data_start = 1997
        print 'year:%d'%year_start
        # ###起止时间计算，生成时间的list，起始时间与97年间隔，持续时间
        year_gap = year_start - data_start  # 6,19
        year_last = year_end - year_start
        year_list = [x + year_gap + 1 + data_start for x in range(year_last)]
        for i in range(len(otherFeatureList)):  ###
            otherFeatureList[i] -= 1
        for i in range(len(baseFeatureList)):  ###
            baseFeatureList[i] += year_gap - 1

    ###所有变量循环
        for k, v in col_name_dict.items():

            ###y标签变换
            labelCol = v + year_gap - 1  # label column number (the year of 1998)



            ###excel读数据，train和test
            oriData = pd.read_excel('../data/data.xlsx', sheetname='Sheet1')
            oriData = oriData.fillna(0.0)
            testData = pd.read_excel('../data/test_data.xlsx')
            testData = testData.fillna(0.0)
            ###相关数据的处理和拆分
            colList = gen_feature_column_numbers(labelCol, baseFeatureList, otherFeatureList)

            train_data_set, test_data_set, col_names = get_cleaned_dataset(oriData, testData, colList, industry_labels)
            label_name = oriData.columns[labelCol].replace('[FY %s]'%(year_start+1), '')
            train_data_set = train_data_set[train_data_set[:, -1] != 0.0]
            test_data_set = test_data_set[test_data_set[:, -1] != 0.0]

            col1=55
            col2=315
            dev_label=[]
            for i in range(1):
                dev_label = []
                for num in range(year_last):
                    if (testData.iloc[i,col1+num]<-10) or (testData.iloc[i,col2+num]<-20) or (testData.iloc[i,col1+num]<0 and testData.iloc[i,col2+num]<-10) or (testData.iloc[i,col1+num]<-5 and testData.iloc[i,col2+num]<-5) :
                        dev_label.append(1)
                    elif (testData.iloc[i,col1+num]>=20 and testData.iloc[i,col2+num]>=0) or  (testData.iloc[i,col1+num]>=10 and testData.iloc[i,col2+num]>=10) :
                        dev_label.append(3)
                    else:
                        dev_label.append(2)
                dev_label.append(round(float(sum(dev_label))/len(dev_label)))
                for num in range(len(dev_label)):
                    print str(dev_label[num])+'\t',
                print
