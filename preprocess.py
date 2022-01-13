import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, dataset):
        self.dataset = dataset

    def getdataset(self):
        return self.dataset

    def get_missing_num(self):
        missing_num = self.dataset.dataset_X.isna().sum
        return missing_num.value().tolist()


    def fillmissingvaluebymedian(self, indexs):
        columns = list(self.dataset.train_set)
        for i in indexs:
            self.dataset.train_set[columns[i]] = self.dataset.train_set[columns[i]].fillna(
                                                 self.dataset.train_set[columns[i]].median())
            self.dataset.test_set[columns[i]] = self.dataset.test_set[columns[i]].fillna(
                                                 self.dataset.train_set[columns[i]].median())
        return self.dataset

    def fillmissingvaluebymean(self, indexs):
        columns = list(self.dataset.train_set)
        for i in indexs:
            self.dataset.train_set[columns[i]] = self.dataset.train_set[columns[i]].fillna(
                                                    self.dataset.train_set[columns[i]].mean())
            self.dataset.test_set[columns[i]] = self.dataset.test_set[columns[i]].fillna(
                self.dataset.test_set[columns[i]].mean())
        return self.dataset

    def fillmissingvaluebymost(self, indexs):
        columns = list(self.dataset.train_set)
        for i in indexs:
            rank_feature = [item for item in self.dataset.train_set[columns[i]].value_counts().index]
            value_most = rank_feature[0]
            self.dataset.train_set[columns[i]] = self.dataset.train_set[columns[i]].fillna(value_most)
            rank_feature = [item for item in self.dataset.test_set[columns[i]].value_counts().index]
            value_most = rank_feature[0]
            self.dataset.test_set[columns[i]] = self.dataset.test_set[columns[i]].fillna(value_most)
        return self.dataset

    def dropcolumns(self, indexs):
        columns = list(self.dataset.train_set)
        for i in indexs:
            self.dataset.train_set = self.dataset.train_set.drop([columns[i]], axis=1)
            self.dataset.test_set = self.dataset.test_set.drop([columns[i]], axis=1)
        return self.dataset

    def getskew(self, indexs):
        #获取选定列的偏斜度
        skew = []
        columns = list(self.dataset.train_set)
        for i in indexs:
            skew.append(self.dataset.train_set[columns[i]].skew())
        return skew

    def getkurt(self, indexs):
        # 获取选定列的峰度
        kurt = []
        columns = list(self.dataset.train_set)
        for i in indexs:
            kurt.append(self.dataset.train_set[columns[i]].kurt())
        return kurt

    def getcorr(self, index1, index2):
        columns = list(self.dataset.train_set)
        column1 = columns[index1]
        column2 = columns[index2]
        return self.dataset.train_set[index1].corr(self.dataset.train_set[index2])

    def floattoint(self, indexs):
        columns = list(self.dataset.train_set)
        for i in indexs:
            self.dataset.train_set[columns[i]] = self.dataset.train_set[columns[i]].astype(int)
            self.dataset.test_set[columns[i]] = self.dataset.test_set[columns[i]].astype(int)

    # TODO数值转换----------------------------------------------------------------

    def minmaxscaler(self):
        scaler = MinMaxScaler()
        scaler.fit(self.dataset.train_set)
        scaler1 = MinMaxScaler()
        scaler1.fit(self.dataset.test_set)

    def standardize(self):
        sc_train = StandardScaler()
        sc_train.fit_transform(self.dataset.train_set)
        sc_test = StandardScaler()
        sc_test.fit_transform(self.dataset.train_set)

