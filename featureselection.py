

class FeatureSelection:
    def __init__(self, dataset):
        self.dataset = dataset

    def manualselect(self, indexs):
        columns = list(self.dataset.train_set)
        selectedcolumns = []
        for i in indexs:
            selectedcolumns.append(columns[i])
        self.dataset.train_set = self.dataset.train_set[selectedcolumns]
        self.dataset.test_set = self.dataset.test_set[selectedcolumns]
        self.dataset.dataset_X = self.dataset.dataset_X[selectedcolumns]

    def corrselect(self, selectnum):
        columns = list(self.dataset.train_set)
        corr = {}
        for item in columns:
            corr[item] = self.dataset.train_label.corr(self.dataset.train_set[item])
        selectcolumns = []
        for i in range(selectnum):
            for k,v in corr:
                if v == max(corr.values()):
                    selectcolumns.append(k)
                    corr[k] = 0
                    break
