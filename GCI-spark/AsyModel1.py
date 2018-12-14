import math, numpy as np
from random import randint
import random
import csv
from typing import Tuple, List

from Classification5 import Classification
import time
from Ensemble1 import Ensemble
from sklearn.decomposition import KernelPCA
import os

BASE_URL = 'datasets/'
SRC_FILE_PATH = os.path.join(BASE_URL, 'norm15group1playertrain.csv')
TGT_FILE_PATH = os.path.join(BASE_URL, 'norm15group1playertest.csv')


class Update(object):
    def __init__(self):
        self.Ensemble = Ensemble(1)
        super(Update, self).__init__()

    @staticmethod
    def readdata(sourcex_matrix=None,
                 sourcey_matrix=None,
                 targetx_matrix=None,
                 targety_matrix=None,
                 src_path=SRC_FILE_PATH,
                 tgt_path=TGT_FILE_PATH,
                 src_size=None,
                 tgt_size=None) -> Tuple[np.matrix, List[int], np.matrix, List[int]]:
        """ 
        input is: source dataset with y, here we assume it is a list of list, the name is source, target dataset with yhat, 
        here we assume it is a list of list, the name is target 
        """
        if sourcex_matrix is None:
            sourcex_matrix_, sourcey_matrix = Classification.read_csv(src_path)  # matrix_ is source data
        else:
            sourcex_matrix_ = sourcex_matrix
            sourcey_matrix_ = sourcey_matrix

        if targetx_matrix is None:
            targetx_matrix_, targety_matrix_ = Classification.read_csv(tgt_path)
        else:
            targetx_matrix_ = targetx_matrix
            targety_matrix_ = targety_matrix

        # get list of all labels
        labelList = []
        for i in range(0, len(targety_matrix_)):
            if targety_matrix_[i] not in labelList:
                labelList.append(targety_matrix_[i])
        print("label list len:", len(labelList))

        # get list of indices of all source y labels
        sourcey_label = []
        for i in range(0, len(sourcey_matrix)):
            sourcey_label.append(labelList.index(sourcey_matrix[i]))

        # get list of indices of all target y labels
        targety_label = []
        for i in range(0, len(targety_matrix_)):
            targety_label.append(labelList.index(targety_matrix_[i]))

        return sourcex_matrix_, sourcey_label, targetx_matrix_, targety_label

    def Process(self,
                sourcex: np.matrix,
                sourcey: List[int],
                targetx: np.matrix,
                targety: List[int],
                subsize: int) -> None:
        # fixed size windows for source stream and target stream
        # kpca = KernelPCA(n_components=3, kernel = 'rbf',gamma = 15)
        # x_kpca = kpca.fit_transform(sourcex)
        # sourcex = np.matrix(x_kpca)
        # y_kpca = kpca.fit_transform(targetx)
        # targetx = np.matrix(y_kpca)

        src_size, _ = sourcex.shape
        tgt_size, _ = targetx.shape

        # get the initial model by using the first source and target windows
        alpha = 0.5
        b = targetx.T.shape[1]
        fold = 5
        sigma_list = Classification.sigma_list(np.array(targetx.T), np.array(sourcex.T))
        lambda_list = Classification.lambda_list()
        # srcx_array = np.array(sourcex.T)
        # trgx_array = np.array(targetx.T)
        # (thetah_old, w, sce_old, sigma_old) = Classification.R_ULSIF(trgx_array, srcx_array, alpha, sigma_list, lambda_list, b, fold)

        # todo
        self.Ensemble.generateNewModelRULSIF(targetx, sourcex, sourcey, alpha, sigma_list,
                                             lambda_list, b, fold, subsize)

        # test model
        trueLableCount = 0.0
        totalCount = 0.0
        for i in range(0, len(targety)):
            instanceResult = self.Ensemble.evaluateEnsembleRULSIF(targetx[i])
            if instanceResult[0] == targety[i]:
                trueLableCount += 1.0
            totalCount += 1.0
            if i % 100 == 0:
                print("test size", i)
                print("true label count", trueLableCount)
                print("total count", totalCount)
                print("tmp acc", trueLableCount / totalCount)
        # write result to file
        with open('output/accuracy_normgroup3datafilegame0726.csv', 'a+') as f:
            writer = csv.writer(f)
            writer.writerow([len(targety), trueLableCount, totalCount, trueLableCount / totalCount])


sourcex_matrix_, sourcey_label_, targetx_matrix_, targety_label_ = Update.readdata()
print("done reading data")

start_time = time.time()
up = Update()
up.Process(sourcex_matrix_, sourcey_label_, targetx_matrix_, targety_label_, 1)
end_time = time.time()
print("Execution time for %d iterations is: %s min" % (1000, (end_time - start_time) / 60.0))
