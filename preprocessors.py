from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

class MultiLabelBinarizerWrapper():
    def __init__(self):
        self.binarizer = MultiLabelBinarizer()
        
    def fit(self, X, y=None):
        self.binarizer.fit(X)
        return self

    def transform(self, X):
        return self.binarizer.transform(X)

    def fit_transform(self, X, y=None):
        return self.binarizer.fit_transform(X)