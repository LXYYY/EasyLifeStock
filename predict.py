import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.datasets import make_classification
from sklearn import preprocessing as prep


class Predictor:
    def __init__(self, features, labels, method="rf"):
        self.method = method
        if self.method == "rf":
            features = np.delete(features, np.where(np.isnan(features))[0], axis=0)
            labels = np.delete(labels, np.where(np.isnan(labels))[0], axis=0)
            self.featuresNorm = prep.scale(features)
            self.labels = labels
            self.estimator = rf(n_estimators=80, max_depth=2, random_state=0)

    def train(self):
        if self.method == "rf":
            self.estimator.fit(self.featuresNorm, self.labels)

    def predict(self, curFeature):
        if self.method == "rf":
            return self.estimator.predict(curFeature)

    def test(self):
        if self.method == "rf":
            featuresTrain = self.featuresNorm[0:int(len(self.featuresNorm) * 0.8), :]
            labelsTrain = self.labels[0:len(featuresTrain) - 1]
            featuresTest = self.featuresNorm[len(featuresTrain):len(self.featuresNorm), :]
            labelsTest = self.labels[len(featuresTrain):len(self.featuresNorm), :]
            if self.method == "rf":
                self.estimator.fit(featuresTrain, labelsTrain)
                labelsRes = self.predict(featuresTest)
                TP = 0
                FP = 0
                TN = 0
                FN = 0
                for i in range(len(labelsTest)):
                    if labelsRes[i] == 1:
                        if labelsRes[i] == labelsTest[i]:
                            TP = TP + 1
                        else:
                            FP = FP + 1
                    if labelsRes[i] == 0:
                        if labelsRes[i] == labelsTest[i]:
                            TN = TN + 1
                        else:
                            FN = FN + 1

                acc = float((TP + TN) / (TP + TN + FP + FN))
                print(acc)
