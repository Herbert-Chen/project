import sys
import os
import random
from string import upper

import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import auc
import matplotlib
matplotlib.style.use('ggplot')


class ArkPlot:
    def __init__(self):
        self.colorList = ['k', 'b', 'r', 'g', 'y', 'c', 'm']

    def line(self, x_batch, y_batch, label_batch, fname, title='', xlabel='', ylabel='', ylim=None, marker=False):
        assert(len(x_batch) == len(y_batch) == len(label_batch))
        plt.figure()
        ii = 0
        for x, y, label in zip(x_batch, y_batch, label_batch):
            assert(len(x)==len(y))
            plt.plot(x, y, '%s-' % self.colorList[ii], label=label)
            if marker:
                plt.plot(x, y, '%s.' % self.colorList[ii])
            ii += 1

        plt.title(title)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim(ylim)
        plt.xticks()
        plt.savefig(fname, format='eps')
        plt.close()

    def histogram_2d(self, x, y, bins, fname, title='', xlabel='', ylabel=''):
        plt.figure()
        plt.hist2d(x, y, bins=bins)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(fname, format='eps')
        plt.close()

    def histogram(self, data, bins, fname, title='', xlabel='', ylabel='', xlim=None):
        plt.figure()
        x,y = np.histogram(data, bins=bins)
        x = 1.0 * x / len(data)
        plt.bar(left=y[:-1],height=x, width=y[1]-y[0])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(y[:-1])
        if fname is None:
            plt.show()
        else:
            plt.savefig(fname, format='eps')
        plt.close()

    def bar(self, data, label, fname, title='', xlabel='', ylabel=''):
        assert(len(data)==len(label))
        n_bars = len(data)
        plt.figure()
        plt.bar(left=np.arange(1,n_bars+1)-0.25, height=data, width=0.5)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(range(1,n_bars+1), label)
        plt.xlim([0, n_bars+1])
        plt.savefig(fname, format='eps')
        plt.close()

    def scatter(self, x, y, fname, title='', xlabel='', ylabel=''):
        plt.figure()
        plt.scatter(x, y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(fname, format='eps')
        plt.close()

    def box(self, data_batch, label_batch, fname, title='', xlabel='', ylabel=''):
        assert(len(data_batch) == len(label_batch))
        plt.figure()
        data_batch = np.array(data_batch).T
        print data_batch.shape, type(data_batch)
        print label_batch

        plt.boxplot(data_batch, labels=label_batch, showfliers=False)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(fname, format='eps')
        plt.close()

    def cdf(self, data_batch, label_batch, fname, title='', xlabel='',
            ylabel='',xlim=[0, 100], fig_width=8, fig_height=8):
        assert(len(data_batch) == len(label_batch))
        plt.figure(figsize=(fig_width,fig_height))

        # determine the max value in the data_batch
        max_value = -1
        for data in data_batch:
            max_value = max(np.max(data), max_value)
        bins = range(0, int(max_value)+1)
        x = np.array(bins[0:-1])

        ii = 0
        plt.plot([0, max_value], [0.5, 0.5], 'k--')
        plt.plot([0, max_value], [0.67, 0.67], 'k--')
        for data, label in zip(data_batch, label_batch):
            hist_value, edge = np.histogram(data, bins=bins)
            hist_value = hist_value * 1.0 / hist_value.sum()
            cdf_value = np.add.accumulate(hist_value)

            median_value = 0
            for idx in xrange(len(cdf_value)):
                if cdf_value[idx] > 0.5:
                    median_value = idx
                    break
            per67_value = 0
            for idx in xrange(len(cdf_value)):
                if cdf_value[idx] >= 0.67:
                    per67_value = idx
                    break

            # plot CDF curve
            #print auc(x[:100], cdf_value[:100])
            plt.plot(x, cdf_value, '%s-' % self.colorList[ii], label=label)

            # plot median line
            plt.plot([bins[median_value], bins[median_value]], [0,1], '%s--' % self.colorList[ii])
            plt.plot([bins[per67_value], bins[per67_value]], [0,1], '%s--' % self.colorList[ii])

            ii += 1

        plt.title(title)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim([0, 1])
        plt.xlim(xlim)
        # plt.xticks([0, 80, 100, 200, 300, 400, 500])
        plt.yticks([0.0,0.2,0.4,0.5,0.6,0.67,0.8,1.0])
        #plt.show()
        plt.savefig(fname, format='png')
        plt.close()


def fig7():
    f_in = '../data/Fig7/'
    f_out = '../img/Fig7/'
    os.system('mkdir -p %s' % f_out)
    data = []
    aplt = ArkPlot()

    ## Variety
    f_log = open(f_out+'variety.txt', 'w')
    f_log.write('mean\tmedian\t%67\n')
    for i in xrange(1, 4):
        print 'load file %d ...' % i
        d = [float(line.strip().split(',')[1]) for line in open(f_in + 'error_exp%d.txt' % i)]
        mean = np.mean(d)
        median = np.median(d)
        d = sorted(d)
        per67 = d[int(len(d)*0.67)]
        f_log.write('%.2f\t%.2f\t%.2f\n' % (mean, median, per67))
        data.append(d)
    label = ['benchmark', 'one-layer model', 'two-layer model']
    params = {
        'data_batch' : data,
        'label_batch' : label,
        'fname' : f_out + 'error_variety.eps',
        'title' : 'variety error',
        'xlabel' : 'error/m',
        'ylabel' : 'percentage',
        'xlim' : [0, 500]
    }
    aplt.cdf(**params)

    ## Interpolation
    f_log = open(f_out+'interpolation.txt', 'w')
    f_log.write('Mean\tMedian\t%67\n')
    mode = ['nointerpolation', 'interpolation_10s', 'median']
    data = []
    for m in mode:
        d = [float(line.strip().split(',')[1]) for line in open(f_in + 'error_%s.txt' % m)]
        mean = np.mean(d)
        median = np.median(d)
        d = sorted(d)
        per67 = d[int(len(d)*0.67)]
        f_log.write('%.2f\t%.2f\t%.2f\n' % (mean, median, per67))
        data.append(d)
    label = ['No Interpolation', 'Interpolation with Interval 10s', 'Interpolation with Interval 2s']
    params = {
        'data_batch' : data,
        'label_batch' : label,
        'fname' : f_out + 'error_interpolation.eps',
        'title' : 'Interpolation Error',
        'xlabel' : 'Error/m',
        'ylabel' : 'Percentage',
        'xlim' : [0, 1000]
    }
    aplt.cdf(**params)


if __name__ == '__main__':
    pass
