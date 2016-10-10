#encoding=utf-8
###依赖包


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from sklearn import cross_validation
from plot import ArkPlot
import sys
from scipy.stats import pearsonr
from matplotlib import pyplot
reload(sys)
sys.setdefaultencoding('utf-8')






#####修改变量


###起始偏移量，1997年开始算，持续时间
# data_start=2010
# year_start=2010
# year_end=2015
###起止时间计算，生成时间的list，起始时间与97年间隔，持续时间         ##########此处需要规范化
# year_gap = year_start - data_start  # 6,19
# year_last = year_end - year_start
# year_list = [x + year_gap + 1 + data_start for x in range(year_last)]

###算法，随机森林，GBRT
algo = 'gbrt'   # rf, gbrt


###路径，使用前一定要检查文件夹是否存在不然报错
var_name='Y9'         ###变量名，Y1，Y2...
folder_name='../result/0827/'+var_name+'/'      ###文件夹名,最后一定要包含/

###文件中定义的格式：
#   =  folder_name + 'cv'+'_'+ var_name + '_' + label'+str(industry_label) + '_' + algo + '_start_' + str(year_start) + '_end_' + str(year_end) + '.png'

###标签,为list，判断是否存在其中，以后如果有多个标签可以扩展为dict
# industry_labels=2

###生成路径，function为 cv,contrast,test...,type为png，txt
def gen_path(function,type):
    return folder_name + function +'_'+ var_name  + '_label_'+ str(industry_labels) + '_' + algo + '_from_' + str(year_start) + '_to_' + str(year_end) + '.'+ type



###y变量，所有变量列号表示，均为excel表中查到的列号，不做任何处理，随时间加减，list中减1均在后面处理,          主函数中
# col_name_dict={'∆Total Cash & ST Investments':111,'∆Total Cash & ST Investments/ Rev %':130}
# col_name_dict={'∆Total Cash & ST Investments':111}
###x特征，分为单一变量和随时间变量，规则同上，处理直接在下面
###此为全部变量
otherFeatureList = [9,13,16]                 ###删掉12,13,12在后面有，13为标签
baseFeatureList = [39,57,75,93,149,168,187,207,225,243,261,279,297,315,333,351]

#削减变量Y2
# otherFeatureList = [23, 25]
# baseFeatureList = [56, 276, 256, 496, 556, 416, 136, 296, 336, 436, 856, 516, 156 ]

#削减变量Y3
# otherFeatureList = [23, 25]
# baseFeatureList = [56, 276, 416, 136, 296, 256, 116, 156, 496, 336, 556, 436, 856, 516]

#削减变量Y5
# otherFeatureList = [22,23, 25]
# baseFeatureList = [256, 416, 276, 56,336,156,136,496,556,296,516,896]


# #削减变量Y6
# otherFeatureList = [23, 25]
# baseFeatureList = [256, 416, 276, 56, 76, 496, 436, 296, 156, 136, 336, 556, 896, 516 ]

#削减变量Y7
# otherFeatureList = [23, 25]
# baseFeatureList = [256, 416, 276, 56, 76, 496, 436, 296, 156, 136, 116, 556, 896, 516 ]






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
        tmpArray = np.array(df_train[indexes].values)
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

    for year_start,year_end in [(2010,2015),(2007,2010)]:#[(1998,2015),(2010,2015),(1999,2002),(2007,2010)]
        if var_name=='Y8':
            col_name_dict = {'∆Total Cash & ST Investments': 111}
            # otherFeatureList = [9, 13, 16]  ###删掉12,13,12在后面有，13为标签
            # baseFeatureList = [39, 57, 75, 93, 149, 168, 187, 207, 225, 243, 261, 279, 297, 315, 333, 351]
        elif var_name=='Y9':
            col_name_dict = {'∆Total Cash & ST Investments/ Rev %':130}
            # otherFeatureList = [9, 13, 16]  ###删掉12,13,12在后面有，13为标签
            # baseFeatureList = [39, 57, 75, 93, 149, 168, 187, 207, 225, 243, 261, 279, 297, 315, 333, 351]

        data_start = 1998
        print 'year:%d-%d'%(year_start,year_end)
        # ###起止时间计算，生成时间的list，起始时间与97年间隔，持续时间
        year_gap = year_start - data_start  # 6,19
        year_last = year_end - year_start
        year_list = [x + year_gap + 1 + data_start for x in range(year_last)]
        # for i in range(len(otherFeatureList)):  ###
        #     otherFeatureList[i] -= 1
        # for i in range(len(baseFeatureList)):  ###
        #     baseFeatureList[i] += year_gap - 1


        for industry_labels in [[],2]:
    ###所有变量循环
            for k, v in col_name_dict.items():
                # if year_start==1998:
                #     otherFeatureList = [16]  ###删掉12,13,12在后面有，13为标签
                #     baseFeatureList = [39, 57, 75, 93, 149, 207, 261, 279, 297, 315, 333, 351]
                # elif industry_labels==[]:
                #     otherFeatureList = [9, 13, 16]  ###删掉12,13,12在后面有，13为标签
                #     baseFeatureList = [39, 57, 75, 93, 149, 207, 261, 279, 297, 315, 333, 351]
                # elif industry_labels==2:
                #     otherFeatureList = [13, 16]  ###删掉12,13,12在后面有，13为标签
                #     baseFeatureList = [39, 57, 75, 93, 149, 207, 261, 279, 297, 315, 333, 351]
                baseFeatureList = [369]
                for i in range(len(otherFeatureList)):  ###
                    otherFeatureList[i] -= 1
                for i in range(len(baseFeatureList)):  ###
                    baseFeatureList[i] += year_gap - 1

                ###y标签变换
                labelCol = v + year_gap - 1  # label column number (the year of 1998)



                ###excel读数据，train和test
                # oriData = pd.read_excel('../data/data.xlsx', sheetname='Sheet1')
                oriData = pd.read_excel('../data/delta_data.xlsx', sheetname='Sheet1')
                oriData = oriData.fillna(0.0)
                testData = pd.read_excel('../data/test_data.xlsx')
                testData = testData.fillna(0.0)
                # colList = gen_feature_column_numbers(labelCol, baseFeatureList, otherFeatureList)
                if industry_labels!=[]:
                    oriData=oriData[ oriData['Industry Dummy'] == industry_labels]
                else: pass
                yName = oriData.columns[labelCol]
                colNames = oriData.columns[baseFeatureList]

                colList = gen_pearson_column_numbers(labelCol, baseFeatureList)
                oriData = oriData.iloc[:,colList]


                data_matrix = oriData.values
                feature_num = len(baseFeatureList)
                for row in xrange(data_matrix.shape[0]):
                    y = data_matrix[row, -year_last:]
                    start = 0
                    for i in xrange(feature_num):
                        end = start + year_last
                        x = data_matrix[row, start:end]
                        start = end
                        print("%f\t" % (pearsonr(x, y)[0])),
                    print
                nameList = []
                # for name in colNames:
                #     nameList.append(name.replace('[FY %s]'%(year_start), '').replace('[FY%s]'%year_start, '').strip())

                # print '\t'.join(nameList)
                # print yName.replace('[FY %s]'%(year_start+1), '')
                print '\n\n'
