#encoding=utf-8
###依赖包


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from sklearn import cross_validation
# from plot import ArkPlot
import sys
from scipy.stats import pearsonr
from matplotlib import pyplot
reload(sys)
sys.setdefaultencoding('utf-8')
algo = 'gbrt'
predict_month=['201512','201612']


###两条路径，一个行数
train_year=3

def transform(df):
    data_and_label=[]
    for i in range(0,172):
        data_and_label.append([])
        for j in range(1,len(df.columns)-train_year):
            flag = True
            for k in range(train_year+1):               ##训练和标签都不包含预测月
                if str(df.columns[j+k]) in predict_month:
                    flag = False
                    # continue
            if flag == True:
                data_and_label[i].extend([df.iloc[i,j:(j+train_year+1)].values])
    return data_and_label

def compute_error(true_y, pred_y):
    error = []
    for i in xrange(true_y.shape[0]):
        error.append(abs(true_y[i] - pred_y[i]))

    return np.array(error)


oriData = pd.read_excel('../data/A3.xlsx', sheetname='Sheet1')
oriData = oriData.fillna(0.0)
train_and_label_data=[]
train_and_label_data=transform(oriData)





###对每个公司：
logger_file_name = '../result/A3cv_train_year_'+str(train_year)+ '.txt'
logger = open(logger_file_name, 'w')
for i in range(0,172):
    train_data = np.array(train_and_label_data)[i][:, :-1]
    label_data = np.array(train_and_label_data)[i][:, -1]
    rs = cross_validation.ShuffleSplit(train_data.shape[0], n_iter=5, test_size=0.1, train_size=0.9,
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
        train_x = train_data[train_index, :]
        train_y = label_data[train_index]
        test_x = train_data[test_index, :]
        test_y = label_data[test_index]

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


    logger.write("company name: %s\n" % (i+1))
    logger.write("\tAverage Min Error: \t%f\n" % (sum(min_error) / len(min_error)))
    logger.write("\tAverage Mean Error: \t%f\n" % (sum(mean_error) / len(mean_error)))
    logger.write("\tAverage Median Error: \t%f\n" % (sum(median_error) / len(median_error)))
    logger.write("\tAverage Max Error: \t%f\n" % (sum(max_error) / len(max_error)))
    logger.write("\n")
logger.close()