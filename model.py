import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

class model:
    def __init__(self):
        self.model = Ridge(0.1)

    def getmodel(self):
        return self.model

    def ridge(self, alpha=0.1):
        self.model = Ridge(alpha=alpha)

    def linear(self):
        self.model = LinearRegression()

    def logistic(self, penalty='l2', solver='lbfgs'):
        self.model = LogisticRegression(penalty=penalty,solver=solver)

    def decisiontree(self, criterion='gini', random_state=None, max_depth=100, min_samples_leaf=1, min_samples_split=1):
        self.model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=random_state,
                                            min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)

    def kneighbors(self, k=5):
        self.model = KNeighborsClassifier(n_neighbors=k)

    def randomforest(self, estimators=100, max_depth=10):
        self.model = RandomForestClassifier(estimators=estimators,max_depth=max_depth)

    def xgboost(self, learning_rate=0.1, n_estimators=160, max_depth=4, min_child_weight=1, gamma=0.0, subsample=0.9,
                colsample_bytree=0.8, objective='binary:logistic', nthread=8, scale_pos_weight=1, seed=27, reg_alpha=0.1):
        self.model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
                                   min_child_weight=min_child_weight, gamma=gamma, subsample=subsample,
                                   colsample_bylevel=colsample_bytree, objective=objective, nthread=nthread,
                                   scale_pos_weight=scale_pos_weight, seed=seed, reg_alpha=reg_alpha)