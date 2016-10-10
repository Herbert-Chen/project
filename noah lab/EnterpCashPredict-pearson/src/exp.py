from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from sklearn import cross_validation
from plot import ArkPlot
import sys
from scipy.stats import pearsonr

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


def gen_pearson_column_numbers(labelCol, baseFeatureList):
    colNeedIndexes = []
    for index in baseFeatureList:
        colNeedIndexes.extend([index + i for i in xrange(19)])
    colNeedIndexes.extend([labelCol + j for j in xrange(19)])
    return colNeedIndexes


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
    labelCol = 455  # label column number (the year of 1998)
    # position that years features first occur
    baseFeatureList = [34, 54, 74, 94, 114, 134, 154, 174, 194, 214, 234, 254, 274,
                       294, 314, 334, 354, 374, 394, 414, 434, 474, 494, 514, 534,
                       554, 594, 614, 634, 654, 794, 814, 834, 854, 874, 894, 914]
    #oriData = pd.read_excel('../data/data.xlsx', sheetname='Sheet1')
    oriData = pd.read_excel('../data/test_data.xlsx')
    oriData = oriData.fillna(0.0)
    yName = oriData.columns[labelCol]
    colNames = oriData.columns[baseFeatureList]

    colList = gen_pearson_column_numbers(labelCol, baseFeatureList)
    oriData = oriData[colList]

    data_matrix = oriData.values
    feature_num = len(baseFeatureList)
    for row in xrange(data_matrix.shape[0]):
        y = data_matrix[row, -19:]
        start = 0
        for i in xrange(feature_num):
            end = start + 19
            x = data_matrix[row, start:end]
            start = end
            print("%f\t" % (pearsonr(x, y)[0])),
        print
    nameList = []
    for name in colNames:
        nameList.append(name.replace('[FY 1997]', '').replace('[FY1997]', '').strip())

    print '\t'.join(nameList)
    print yName.replace('[FY 1998]', '')








