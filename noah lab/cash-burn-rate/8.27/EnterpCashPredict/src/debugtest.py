#encoding=utf-8
###依赖包


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from sklearn import cross_validation
# from plot import ArkPlot
import sys
from matplotlib import pyplot
from scipy.stats import pearsonr
reload(sys)
sys.setdefaultencoding('utf-8')






#####修改变量


###起始偏移量，1997年开始算，持续时间
# year_start=1997
# year_end=2016
# data_start=1998
###起止时间计算，生成时间的list，起始时间与97年间隔，持续时间         ##########此处需要规范化
# year_gap = year_start - data_start  # 6,19
# year_last = year_end - year_start
# year_list = [x + year_gap + 1 + data_start for x in range(year_last)]

###算法，随机森林，GBRT
algo = 'gbrt'   # rf, gbrt


###路径，使用前一定要检查文件夹是否存在不然报错
var_name='Y7'         ###变量名，Y1，Y2...
folder_name='../result/1021/'+var_name+'/'      ###文件夹名,最后一定要包含/

###文件中定义的格式：
#   =  folder_name + 'cv'+'_'+ var_name + '_' + label'+str(industry_label) + '_' + algo + '_start_' + str(year_start) + '_end_' + str(year_end) + '.png'

###标签,为list，判断是否存在其中，以后如果有多个标签可以扩展为dict
# industry_labels=2

###生成路径，function为 cv,contrast,test...,type为png，txt
# def gen_path(function,type):
#     return folder_name + function +'_'+ var_name  + '_industry_label_'+ str(industry_labels) + '_dev_label_'+str(dev_label) +'_'+ algo + '_from_' + str(year_start) + '_to_' + str(year_end) + '.'+type



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
def get_cleaned_dataset(df_train, df_test,colsList, industry_label,dev_label):
    if industry_label !=[]:
        # df_train = df_train[ df_train['Industry Dummy'] in industry_label]
        df_train = df_train[df_train['Industry Dummy'] == industry_label]       ###扩展！！！！！
    else:
        pass
    col_names = df_train.columns[colsList[0]]
    count = 0
    if dev_label !=[]:
        # df_train = df_train[ df_train['Industry Dummy'] in industry_label]
        df_train = df_train[df_train['Dev_label'] == dev_label]       ###扩展！！！！！
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
        testTmpArray = np.array(df_test.iloc[:,indexes].values)
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
    for var_name in ['Y2','Y3','Y5','Y6','Y7']:
        folder_name = '../result/1021/' + var_name + '/'
        def gen_path(function, type):
            return folder_name + function + '_' + var_name + '_industry_label_' + str(
                industry_labels) + '_dev_label_' + str(dev_label) + '_' + algo + '_from_' + str(
                year_start) + '_to_' + str(year_end) + '.' + type


        for year_start in [1997,2010]:
            if var_name == 'Y2':
                # 削减变量Y2
                col_name_dict = {'Total Cash & ST Investments/Total Revenue %': 576}
                otherFeatureList = [12, 25,993]
                baseFeatureList = [56, 136, 156, 256, 276, 296, 336, 416, 436, 496, 516, 556, 896]
            elif var_name == 'Y3':
                # 削减变量Y3
                col_name_dict = {'Total Cash & ST Investments/Avg. monthly cash outflow Month': 676}
                otherFeatureList = [12, 25,993]
                baseFeatureList = [56, 116, 136, 156, 256, 276, 296, 336, 416, 436, 496, 516, 556, 896]
            elif var_name == 'Y5':
                # 削减变量Y5
                col_name_dict = {'Total Cash & ST Investments/Total Assets %': 736}
                otherFeatureList = [22, 12, 25,993]
                baseFeatureList = [56, 136, 156, 256, 276, 296, 336, 416, 496, 516, 556, 896]
            elif var_name == 'Y6':
                # #削减变量Y6
                col_name_dict = {'Net Cash Burn Ratio 1=Net Cash 1/ Avg. monthly cash outflow %': 756}
                otherFeatureList = [12, 25,993]
                baseFeatureList = [56, 76, 136, 156, 256, 276, 296, 336, 416, 436, 496, 516, 556, 896]
            elif var_name == 'Y7':
                # 削减变量Y7
                col_name_dict = {'Net Cash 1 / Sales %': 776}
                otherFeatureList = [12, 25,993]
                baseFeatureList = [56, 76, 116, 136, 156, 256, 276, 296, 416, 436, 496, 516, 556, 896]
            #     ###x标签变换
            # otherFeatureList = [ 22, 23, 24, 25,27,993]  # unique features
            # # position that years features first occur
            # baseFeatureList = [34, 54, 74, 94, 114, 134, 154, 174, 194, 214, 234, 254, 274,
            #                    294, 314, 334, 354, 374, 394, 414, 434, 474, 494, 514, 534,
            #                    554, 594, 614, 634, 654, 794, 814, 834, 854, 874, 894, 914]
            year_end = 2016
            data_start = 1997
            # print 'year:%d'%year_start
            # ###起止时间计算，生成时间的list，起始时间与97年间隔，持续时间
            year_gap = year_start - data_start  # 6,19
            year_last = year_end - year_start
            year_list = [x + year_gap + 1 + data_start for x in range(year_last)]
            for i in range(len(otherFeatureList)):  ###
                otherFeatureList[i] -= 1
            for i in range(len(baseFeatureList)):  ###
                baseFeatureList[i] += year_gap - 1


            for industry_labels in [[],0,1,2]:
        ###所有变量循环
                for dev_label in [2,3]:

                    for k, v in col_name_dict.items():

                        print var_name+'_industrylabel_'+str(industry_labels)+'_devlabel_'+str(dev_label)+'_'+str(year_start)+'-'+str(year_end)

                        ###y标签变换
                        labelCol = v + year_gap - 1  # label column number (the year of 1998)



                        ###excel读数据，train和test
                        oriData = pd.read_excel('../data/data.xlsx', sheetname='Sheet1')
                        oriData = oriData.fillna(0.0)
                        testData = pd.read_excel('../data/test_data.xlsx')
                        testData = testData.fillna(0.0)
                        ###相关数据的处理和拆分
                        colList = gen_feature_column_numbers(labelCol, baseFeatureList, otherFeatureList)

                        train_data_set, test_data_set, col_names = get_cleaned_dataset(oriData, testData, colList, industry_labels,dev_label)
                        label_name = oriData.columns[labelCol].replace('[FY %s]'%(year_start+1), '')
                        train_data_set = train_data_set[train_data_set[:, -1] != 0.0]
                        test_data_set = test_data_set[test_data_set[:, -1] != 0.0]

                        train_feature_set, train_label_set = split_feature_label(train_data_set)
                        test_feature_set, test_label_set = split_feature_label(test_data_set)

                        ###输出维度
                        # print 'train_feature_set:%s'%str(train_feature_set.shape)
                        # print 'train_label_set:%s' % str(train_label_set.shape)
                        # print 'test_feature_set:%s' % str(test_feature_set.shape)
                        # print 'test_label_set:%s' % str(test_label_set.shape)


                        ###交叉验证绘图
                        plt = ArkPlot()
                        plt_out_name = gen_path('cv','png')
                        rs = cross_validation.ShuffleSplit(train_data_set.shape[0], n_iter=5, test_size=0.1, train_size=0.9,
                                                           random_state=None)  # 10 fold cross validation start

                        # logger_file_name = '../result/1.2/cv_Y2' + algo +  '_gap_' +str(year_gap) + '_last_'+str(year_last) +  '.txt'
                        # logger = open(logger_file_name, 'w')
                        min_error = []
                        mean_error = []
                        median_error = []
                        max_error = []
                        count = 0
                        err_data = []
                        plot_label = []
                        plot_max_error = -1
                        for train_index, test_index in rs:
                            # logger.write("Label: \t%s\n" % label_name)
                            # print count + 1
                            count += 1
                            train_x = train_feature_set[train_index, :]
                            train_y = train_label_set[train_index]
                            test_x = train_feature_set[test_index, :]
                            test_y = train_label_set[test_index]

                            if algo == 'rf':
                                model = RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=None, min_samples_split=2,
                                                              max_features=0.5, n_jobs=-1)
                            elif algo == 'gbrt':
                                model = GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=1000,
                                                                  min_samples_split=2,
                                                                  max_depth=5, max_features='sqrt', alpha=0.9)

                            model.fit(train_x, train_y)
                            pred_y = model.predict(test_x)

                            error = compute_error(test_y, pred_y)
                            plot_max_error = max(np.max(error), plot_max_error)
                            err_sorted = sorted(error)
                            err_data.append(err_sorted)
                            plot_label.append('cv' + str(count))

                            min_error.append(np.min(error))
                            mean_error.append(np.mean(error))
                            median_error.append(np.median(error))
                            max_error.append(np.max(error))

                            label = ['benchmark', 'one-layer model', 'two-layer model']
                            params = {
                                'data_batch': err_data,
                                'label_batch': plot_label,
                                'fname': plt_out_name,
                                'title': label_name,
                                'xlabel': 'error',
                                'ylabel': 'percentage',
                                # 'xlim': [0, plot_max_error * 0.5]
                                'xlim': [0, 160]
                            }
                            plt.cdf(**params)

                        ###交叉验证的文档，最大最小。。。的平均
                        logger_file_name = gen_path('cv', 'txt')
                        logger = open(logger_file_name, 'w')
                        logger.write("\tLabel: %s\n" % label_name)
                        logger.write("\tAverage Min Error: \t%f\n" % (sum(min_error)/len(min_error)))
                        logger.write("\tAverage Mean Error: \t%f\n" % (sum(mean_error)/len(mean_error)))
                        logger.write("\tAverage Median Error: \t%f\n" % (sum(median_error)/len(median_error)))
                        logger.write("\tAverage Max Error: \t%f\n" % (sum(max_error)/len(max_error)))
                        logger.close()

                        print '交叉验证'
                        print "\tLabel: %s" % label_name
                        print "\tAverage Min Error: \t%f" % (sum(min_error)/len(min_error))
                        print "\tAverage Mean Error: \t%f" % (sum(mean_error)/len(mean_error))
                        print "\tAverage Median Error: \t%f" % (sum(median_error)/len(median_error))
                        print "\tAverage Max Error: \t%f" % (sum(max_error)/len(max_error))
                        print
                        ###HW上验证

                        # print train_data_set.shape

                        min_error = []
                        mean_error = []
                        median_error = []
                        max_error = []

                        err_data = []
                        plot_label = []
                        plot_max_error = -1


                        if algo == 'rf':
                            model = RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=None, min_samples_split=2,
                                                          max_features=0.5, n_jobs=-1)
                        elif algo == 'gbrt':
                            model = GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=1000, min_samples_split=2,
                                                              max_depth=5, max_features='sqrt', alpha=0.9)

                        model.fit(train_feature_set, train_label_set)
                        pred_y = model.predict(test_feature_set)

                        error = compute_error(test_label_set, pred_y)
                        plot_max_error = max(np.max(error), plot_max_error)
                        err_sorted = sorted(error)
                        err_data.append(err_sorted)
                        plot_label.append('test')
                        min_error.append(np.min(error))
                        mean_error.append(np.mean(error))
                        median_error.append(np.median(error))
                        max_error.append(np.max(error))

                        ranking_index = np.argsort(-model.feature_importances_)
                        importance_ranking_list = model.feature_importances_[ranking_index]
                        feature_ranking_list = col_names[ranking_index]


                        ###写测试log
                        logger_file_name = gen_path('test','txt')
                        logger = open(logger_file_name, 'w')
                        logger.write("\tLabel: %s\n" % label_name)
                        logger.write("\tMin Error: \t%f\n" % np.min(error))
                        logger.write("\tMean Error: \t%f\n" % np.mean(error))
                        logger.write("\tMedian Error: \t%f\n" % np.median(error))
                        logger.write("\tMax Error: \t%f\n" % np.max(error))
                        logger.write('\tFeature importance:\n')
                        for i in xrange(len(ranking_index)):
                            logger.write("\t\t%s: %f\n" % (
                            feature_ranking_list[i].replace('[FY %s]' % year_start, '').replace('[FY%s]' % year_start, ''),
                            importance_ranking_list[i]))
                        logger.write('\n')

                        # log total error indexes
                        logger.write("\tTotal Min: %f\n" % np.mean(np.array(min_error)))
                        logger.write("\tTotal Mean: %f\n" % np.mean(np.array(mean_error)))
                        logger.write("\tTotal Median: %f\n" % np.mean(np.array(median_error)))
                        logger.write("\tTotal Max: %f\n" % np.mean(np.array(max_error)))
                        logger.write("\tprediction and real value:\n" )
                        for i in range(len(pred_y)):
                            logger.write("\t%d:\t%f    %f\n"%(year_start+1+i,float(pred_y[i]),float(test_label_set[i])))
                        logger.close()

                        print '华为上验证：'
                        print "\tMin Error: \t%f" % np.min(error)
                        print "\tMean Error: \t%f" % np.mean(error)
                        print "\tMedian Error: \t%f" % np.median(error)
                        print "\tMax Error: \t%f" % np.max(error)
                        print '\tFeature importance:'
                        for i in xrange(len(ranking_index)):
                            print "\t\t%s: %f" % (
                            feature_ranking_list[i].replace('[FY %s]' % year_start, '').replace('[FY%s]' % year_start, ''),
                            importance_ranking_list[i])
                        print
                        plt_hw = ArkPlot()
                        plt_hw_out_name = gen_path('test','png')

                        params = {
                            'data_batch': err_data,
                            'label_batch': plot_label,
                            'fname': plt_hw_out_name,
                            'title': label_name,
                            'xlabel': 'error',
                            'ylabel': 'percentage',
                            # 'xlim': [0, plot_max_error * 0.2]
                            'xlim': [0, 160]
                        }

                        plt_hw.cdf(**params)

                        # pyplot.figure()，对比预测与实际值
                        pyplot.plot(year_list, pred_y, label='prediction', color='red')
                        pyplot.plot(year_list, test_label_set, label='ground truth', color='black')
                        pyplot.title(k)
                        pyplot.xlabel('year')
                        pyplot.savefig(gen_path('contrast','png'))
                        pyplot.legend()
                        pyplot.close()




                        ###预测16-20
                        future_Data = pd.read_excel('../data/future_'+var_name+'.xlsx', sheetname='Sheet1')
                        pred_future = model.predict(np.array(future_Data.values))
                        # print var_name
                        # print 'year:%d-2015' % year_start
                        # print 'industry_labels' + str(industry_labels)
                        # print 'dev_label' + str(dev_label)
                        # print var_name+'_industrylabel_'+str(industry_labels)+'_devlabel_'+str(dev_label)+'_'+str(year_start)+'-'+str(year_end)
                        print '预测未来：16-20'
                        for num in range(len(pred_future)):
                            print str(pred_future[num]) + '\t',
                    print '\n\n'

            print
        print
