import math, sklearn.metrics.pairwise as sk
from sklearn import svm
import numpy as np
import random, sys

from sklearn.svm import SVC


class Model(object):
    def __init__(self):
        self.model: SVC = None
        self.weight = 0.0

    def test(self, testDataX: np.matrix):
        confidences = []
        predictions = self.model.predict(testDataX)
        probs = self.model.predict_proba(testDataX)
        for i in range(0, len(testDataX)):
            for j in range(len(self.model.classes_)):
                if self.model.classes_[j] == predictions[i]:
                    confidences.append(probs[i][j])
                    break
        return predictions, confidences

    def computeModelWeightRULSIF(self, data: np.matrix) -> float:
        totalConf = 0.0
        predictions, confidences = self.test(data)
        for i in range(0, len(confidences)):
            totalConf += confidences[i]
        return totalConf / len(data)

    def __computeWeight(self, errorRate):
        if errorRate <= 0.5:
            if errorRate == 0:
                errorRate = 0.01
            return 0.5 * math.log((1 - errorRate) / errorRate)
        else:
            return 0.01
