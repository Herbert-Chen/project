#encoding=utf-8
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from sklearn import cross_validation
from plot import ArkPlot
import sys
from matplotlib import pyplot
reload(sys)
sys.setdefaultencoding('utf-8')



###起始偏移量，1997年开始算，持续时间
year_start=1997
year_end=2010
year_gap=year_start-1997  #6,19
year_last=year_end-year_start
year_list=[x+year_gap+1+1997 for x in range(year_last) ]


###加入year变量，后续只记录97，根据所选年份增加。具体计算：开始年份到2015年
###后续有一个replace在100行左右也需要修改









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
def get_cleaned_dataset(df, colsList):
    col_names = df.columns[colsList[0]]
    count = 0
    for indexes in colsList:
        tmpArray = np.array(df[indexes].values)
        if count == 0:
            upperArray = tmpArray
            count += 1
            continue
        else:
            upperArray = np.vstack((upperArray, tmpArray))
            count += 1

    return upperArray, col_names

#######
# Function: merge sub datasets into one
# Input: df: pandas.DataFrame, colsList: list[list[int]]
# Output: upperArray: np.array, col_names: list[string]
#######
def get_cleaned_dataset_testone(df_train, df_test, colsList):
    col_names = df_train.columns[colsList[0]]
    count = 0
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
# Function: merge sub datasets into one
# Input: df: pandas.DataFrame, colsList: list[list[int]]
# Output: upperArray: np.array, col_names: list[string]
#######
def get_cleaned_dataset_by_industry(df, colsList, industry):
    df_filtered = df[df['Industry Dummy'] == industry]
    col_names = df.columns[colsList[0]]
    count = 0
    for indexes in colsList:
        tmpArray = np.array(df_filtered[indexes].values)
        if count == 0:
            upperArray = tmpArray
            count += 1
            continue
        else:
            upperArray = np.vstack((upperArray, tmpArray))
            count += 1

    return upperArray, col_names


#######
# Function: split features and label from dataset
# Input: data_set: np.array
# Output: features: np.array, label: np.array
#######
def split_feature_label(data_set):
    features = data_set[:, :-1]
    label = data_set[:, -1]
    return features, label


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
    algo = 'gbrt'   # rf, gbrt
    col_name_dict={'Total Cash & ST Investments':456,'Total Cash & ST Investments/Total Revenue %':576,'Total Cash & ST Investments/Avg. monthly cash outflow Month':676, \
                   'Net Cash 1 = Total Cash & ST Investments - Debt':716,'Total Cash & ST Investments/Total Assets %':736 ,'Net Cash Burn Ratio 1=Net Cash 1/ Avg. monthly cash outflow %':756, \
                   'Net Cash 1 / Sales %':776}
    ###
    for k,v in col_name_dict.items():
        labelCol = v + year_gap - 1  # label column number (the year of 1998)
        otherFeatureList = [13, 22, 23, 24, 25,27]     # unique features   ###新增cash flow risk，25列,删除12列(11)，直接用23
        for i in range(len(otherFeatureList)):               ###
            otherFeatureList[i]-=1
        # position that years features first occur
        baseFeatureList = [36, 56, 76, 96, 116, 136, 156, 176, 196, 216, 236, 256, 276,
                       296, 316, 336, 356, 376, 396, 416, 436, 476, 496, 516, 536,
                       556, 596, 616, 636, 656, 796, 816, 836, 856, 876, 896, 916]
        for i in range(len(baseFeatureList)):               ###
            baseFeatureList[i]+=year_gap-1

        oriData = pd.read_excel('../data/data.xlsx', sheetname='Sheet1')
        oriData = oriData.fillna(0.0)

        colList = gen_feature_column_numbers(labelCol, baseFeatureList, otherFeatureList)
        data_set, col_names = get_cleaned_dataset(oriData, colList)
        label_name = oriData.columns[labelCol].replace('[FY %s]'%str(year_start+1), '')  ###
        data_set_no_zero = data_set[data_set[:, -1] != 0.0]
        feature_set, label_set = split_feature_label(data_set_no_zero)

        plt = ArkPlot()
        plt_out_name = '../result/1.x/' + 'cv_Y2' + algo + '_gap_' +str(year_gap) + '_last_'+str(year_last) + '.png'        ###

        print data_set_no_zero.shape

        rs = cross_validation.ShuffleSplit(data_set_no_zero.shape[0], n_iter=5, test_size=0.1, train_size=0.9,
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
            print count + 1
            count += 1
            train_x = feature_set[train_index, :]
            train_y = label_set[train_index]
            test_x = feature_set[test_index, :]
            test_y = label_set[test_index]

            if algo == 'rf':
                model = RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=None, min_samples_split=2,
                                       max_features=0.5, n_jobs=-1)
            elif algo == 'gbrt':
                model = GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=1000, min_samples_split=2,
                                                  max_depth=5, max_features='sqrt', alpha=0.9)

            model.fit(train_x, train_y)
            pred_y = model.predict(test_x)

            error = compute_error(test_y, pred_y)
            plot_max_error = max(np.max(error), plot_max_error)
            err_sorted = sorted(error)
            err_data.append(err_sorted)
            plot_label.append('cv' + str(count))

            print np.min(error)
            print np.mean(error)
            print np.median(error)
            print np.max(error)
            # logger.write("Min Error: \t%f\n" % np.min(error))
            # logger.write("Mean Error: \t%f\n" % np.mean(error))
            # logger.write("Median Error: \t%f\n" % np.median(error))
            # logger.write("Max Error: \t%f\n" % np.max(error))
            min_error.append(np.min(error))
            mean_error.append(np.mean(error))
            median_error.append(np.median(error))
            max_error.append(np.max(error))

            # ranking_index = np.argsort(-model.feature_importances_)
            # importance_ranking_list = model.feature_importances_[ranking_index]
            # feature_ranking_list = col_names[ranking_index]

            # logger.write('Feature importance:\n')
            # for i in xrange(len(ranking_index)):
            #     logger.write("\t %s: \t%f\n" % (feature_ranking_list[i].replace('[FY 2010]', '').replace('[FY2010]', ''),
            #                                importance_ranking_list[i]))
            # logger.write('\n')
        # 10 fold cross validation end

        # # log total error indexes
        # logger.write("Total Min: \t%f\n" % np.mean(np.array(min_error)))
        # logger.write("Total Mean: \t%f\n" % np.mean(np.array(mean_error)))
        # logger.write("Total Median: \t%f\n" % np.mean(np.array(median_error)))
        # logger.write("Total Max: \t%f\n" % np.mean(np.array(max_error)))
        # logger.close()

        # plot
        label = ['benchmark', 'one-layer model', 'two-layer model']
        params = {
            'data_batch': err_data,
            'label_batch': plot_label,
            'fname': plt_out_name,
            'title': label_name,
            'xlabel': 'error',
            'ylabel': 'percentage',
            #'xlim': [0, plot_max_error * 0.5]
            'xlim': [0, 160]
        }
        plt.cdf(**params)

        testData = pd.read_excel('../data/test_data.xlsx')
        testData = testData.fillna(0.0)
        train_data_set, test_data_set, col_names = get_cleaned_dataset_testone(oriData, testData, colList)
        label_name = oriData.columns[labelCol].replace('[FY 1998]', '')
        train_data_set = train_data_set[train_data_set[:, -1] != 0.0]
        test_data_set = test_data_set[test_data_set[:, -1] != 0.0]

        train_feature_set, train_label_set = split_feature_label(train_data_set)
        test_feature_set, test_label_set = split_feature_label(test_data_set)

        ###
        plt_hw = ArkPlot()
        plt_hw_out_name = '../result/1.x/' + 'testone_Y2' + '_' + algo + '_gap_' +str(year_gap) + '_last_'+str(year_last) +  '.png'

        print train_data_set.shape

        logger_file_name = '../result/1.x/testone_Y2_' + algo + '_gap_' +str(year_gap) + '_last_'+str(year_last) +  '.txt'
        logger = open(logger_file_name, 'w')
        min_error = []
        mean_error = []
        median_error = []
        max_error = []

        err_data = []
        plot_label = []
        plot_max_error = -1

        logger.write("Label: %s\n" % label_name)

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

        print np.min(error)
        print np.mean(error)
        print np.median(error)
        print np.max(error)
        logger.write("Min Error: \t%f\n" % np.min(error))
        logger.write("Mean Error: \t%f\n" % np.mean(error))
        logger.write("Median Error: \t%f\n" % np.median(error))
        logger.write("Max Error: \t%f\n" % np.max(error))
        min_error.append(np.min(error))
        mean_error.append(np.mean(error))
        median_error.append(np.median(error))
        max_error.append(np.max(error))

        ranking_index = np.argsort(-model.feature_importances_)
        importance_ranking_list = model.feature_importances_[ranking_index]
        feature_ranking_list = col_names[ranking_index]

        logger.write('Feature importance:\n')
        for i in xrange(len(ranking_index)):
            logger.write("\t%s: %f\n" % (feature_ranking_list[i].replace('[FY %s]'%year_start, '').replace('[FY%s]'%year_start, ''),
                                         importance_ranking_list[i]))
        logger.write('\n')

        # log total error indexes
        logger.write("Total Min: %f\n" % np.mean(np.array(min_error)))
        logger.write("Total Mean: %f\n" % np.mean(np.array(mean_error)))
        logger.write("Total Median: %f\n" % np.mean(np.array(median_error)))
        logger.write("Total Max: %f\n" % np.mean(np.array(max_error)))
        logger.close()

        # plot
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

        # pyplot.figure()
        pyplot.plot(year_list, pred_y, label='prediction', color='red')
        pyplot.plot(year_list, test_label_set, label='ground truth', color='black')
        pyplot.title(k)
        pyplot.xlabel('year')
        pyplot.savefig('../result/1.x/' + 'contrast' + '_Y2_' + algo + '_gap_' +str(year_gap) + '_last_'+str(year_last) +  '.png')
        pyplot.legend()
        pyplot.close()



