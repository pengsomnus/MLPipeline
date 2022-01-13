import pandas as pd
import numpy as np
import os


class DataLoader:
    # 数据读取类，参数为路径，和测试集大小
    def __init__(self, name, path, test_ratio, labelindex):
        self.name = name
        dataset = self.datasetloader(path)
        columns = list(dataset)
        self.dataset_label = dataset[columns[labelindex]]
        print(columns[labelindex])
        self.dataset_X = dataset.drop(columns[labelindex], axis=1)
        self.train_set, self.test_set = self.split_train(dataset, test_ratio)
        self.train_label = self.train_set[columns[labelindex]]
        self.train_set.drop(columns[labelindex], axis=1)
        self.test_label = self.test_set[columns[labelindex]]
        self.test_set.drop(columns[labelindex], axis=1)
        self.missing_total = self.train_set.isnull().sum().sort_values(ascending=False)

    def datasetloader(self, path):
        if os.path.exists(path):
            str = ".xlsx"
            print(path)
            if str in path:
                dataset = pd.read_excel(path)
            else:
                dataset = pd.read_csv(path)
            return dataset

    def split_train(self, dataset, test_ratio, random_seed=43):
        np.random.seed(random_seed)
        shuffled_indices = np.random.permutation(len(dataset))
        test_set_size = int(len(dataset)*test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return dataset.iloc[train_indices], dataset.iloc[test_indices]
