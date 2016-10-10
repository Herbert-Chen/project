#coding=utf-8
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation
import numpy as np
import pandas as pd
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

months = ['201201', '201202', '201203', '201204', '201205', '201206', '201207', '201208', '201209', '201210',
          '201211', '201212', '201301', '201302', '201303', '201304', '201305', '201306', '201307', '201308',
          '201309', '201310', '201311', '201312', '201401', '201402', '201403', '201404', '201405', '201406',
          '201407', '201408', '201409', '201410', '201411', '201412', '201501', '201502', '201503', '201504',
          '201505', '201506', '201507', '201508', '201509', '201510', '201511', '201512', '201601', '201602',
          '201603', '201604', '201605']


def get_features_and_label(df, last_label_month, ahead_month):
    data = df.values
    label_row = months.index(last_label_month)
    step = int(ahead_month)
    feature_row = label_row - step

    feature_row_index = []
    label_row_index = []
    while feature_row >= 0:
        feature_row_index.append(feature_row)
        label_row_index.append(label_row)
        feature_row -= 1
        label_row -= 1

    features = data[feature_row_index, 1:64]
    labels = data[label_row_index, 63]

    return features, labels


def get_predict_feature(df, predict_feature_month):
    data = df.values
    feature_row = months.index(predict_feature_month)
    return data[feature_row, 1:64]


def compute_error(true_y, pred_y):
    error = []
    for i in xrange(true_y.shape[0]):
        error.append(abs(true_y[i] - pred_y[i]))

    return np.array(error)


def run_cross_validation(model, x, y, n_iter):
    rs = cross_validation.ShuffleSplit(x.shape[0], n_iter=n_iter, test_size=0.25, train_size=0.75,
                                       random_state=None)  # cross validation start

    logger = open('cv_report.txt', 'w')
    min_error = []
    mean_error = []
    median_error = []
    max_error = []
    count = 0
    err_data = []
    plot_label = []
    plot_max_error = -1
    for train_index, test_index in rs:
        #logger.write("Label: %s\n" % label_name)
        print count + 1
        count += 1
        train_x = x[train_index, :]
        train_y = y[train_index]
        test_x = x[test_index, :]
        test_y = y[test_index]

        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)

        error = compute_error(test_y, pred_y)
        #plot_max_error = max(np.max(error), plot_max_error)
        #err_sorted = sorted(error)
        #err_data.append(err_sorted)
        #plot_label.append('cv' + str(count))

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

if __name__ == '__main__':
    algo = 'gbrt'   # rf, gbrt
    df = pd.read_csv('../data/cash_flow_data.csv', encoding='utf8')

    
    # label month means the last label month for current obvious data.
    label_month_list = ['201505', '201506', '201507', '201508', '201509', '201510', '201511', '201512',
                        '201601', '201602', '201603', '201604']

    # moving on every current last label month from 201505 to 201604
    for label_month in label_month_list:
        # different max ahead month for each month
        max_ahead_month = months.index('201605') - months.index(label_month)
        
        '''
        Extract predict feature. example: current month is 201506, 
        so the last label month is 201505 for training model and 
        would extract 201505 as the feature to predict future month.
        '''
        predict_feature = get_predict_feature(df, label_month)
        print 'label_month: %s\tmax_ahead_month: %s' % (label_month, str(max_ahead_month))
        print 'Predict result (ahead month from 1 to max_ahead_month):'
        for ahead_month in range(1, max_ahead_month + 1):
            x, y = get_features_and_label(df, label_month, ahead_month)

            if algo == 'rf':
                model = RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=None, min_samples_split=2,
                                       max_features='sqrt', n_jobs=-1)
            elif algo == 'gbrt':
                model = GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=1000,
                                                  min_samples_split=2,
                                                  max_depth=5, max_features='sqrt', alpha=0.9)

                model.fit(x, y)

            print '%.2f' % model.predict(predict_feature.reshape(1, -1))

    #run_cross_validation(rf, x, y, 10)



