from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from sklearn import cross_validation
from plot import ArkPlot
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


#######
# Function: Generate column numbers of each subset (by year)
# Input: labelCol: int, baseFeatureList: list[int], otherFeatureList: list[int]
# Output: featureLabelCols: list[list[int]]
#######
def gen_feature_column_numbers(labelCol, baseFeatureList, otherFeatureList):
    featureLabelCols = []
    for step in xrange(19):
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
    labelCol = 775  # label column number (the year of 1998)
    otherFeatureList = [11, 12, 21, 22, 23, 25]     # unique features
    # position that years features first occur
    baseFeatureList = [34, 54, 74, 94, 114, 134, 154, 174, 194, 214, 234, 254, 274,
                       294, 314, 334, 354, 374, 394, 414, 434, 474, 494, 514, 534,
                       554, 594, 614, 634, 654, 794, 814, 834, 854, 874, 894, 914]
    oriData = pd.read_excel('../data/data.xlsx', sheetname='Sheet1')
    oriData = oriData.fillna(0.0)

    colList = gen_feature_column_numbers(labelCol, baseFeatureList, otherFeatureList)
    data_set, col_names = get_cleaned_dataset(oriData, colList)
    label_name = oriData.columns[labelCol].replace('[FY 1998]', '')
    data_set_no_zero = data_set[data_set[:, -1] != 0.0]
    feature_set, label_set = split_feature_label(data_set_no_zero)

    plt = ArkPlot()
    plt_out_name = '../figures/' + 'cv_' + algo + '_' + str(labelCol) + '.png'

    print data_set_no_zero.shape

    rs = cross_validation.ShuffleSplit(data_set_no_zero.shape[0], n_iter=5, test_size=0.1, train_size=0.9,
                                        random_state=None)  # 10 fold cross validation start

    logger_file_name = '../result/cv_' + str(labelCol) + '.txt'
    logger = open(logger_file_name, 'w')
    min_error = []
    mean_error = []
    median_error = []
    max_error = []
    count = 0
    err_data = []
    plot_label = []
    plot_max_error = -1
    for train_index, test_index in rs:
        logger.write("Label: %s\n" % label_name)
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
        logger.write("Min Error: %f\n" % np.min(error))
        logger.write("Mean Error: %f\n" % np.mean(error))
        logger.write("Median Error: %f\n" % np.median(error))
        logger.write("Max Error: %f\n" % np.max(error))
        min_error.append(np.min(error))
        mean_error.append(np.mean(error))
        median_error.append(np.median(error))
        max_error.append(np.max(error))

        ranking_index = np.argsort(-model.feature_importances_)
        importance_ranking_list = model.feature_importances_[ranking_index]
        feature_ranking_list = col_names[ranking_index]

        logger.write('Feature importance:\n')
        for i in xrange(len(ranking_index)):
            logger.write("%s: %f\n" % (feature_ranking_list[i].replace('[FY 1997]', '').replace('[FY1997]', ''),
                                       importance_ranking_list[i]))
        logger.write('\n')
    # 10 fold cross validation end

    # log total error indexes
    logger.write("Total Min: %f\n" % np.mean(np.array(min_error)))
    logger.write("Total Mean: %f\n" % np.mean(np.array(mean_error)))
    logger.write("Total Median: %f\n" % np.mean(np.array(median_error)))
    logger.write("Total Max: %f\n" % np.mean(np.array(max_error)))
    logger.close()

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





