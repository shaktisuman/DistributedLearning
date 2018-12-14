import csv
from typing import Tuple, List

import numpy as np
from pylab import *
from scipy import linalg
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import time
import logging

from MatrixOperation import fast_matmul

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=None,
                    level=logging.INFO)
logger = logging.getLogger('Baseline')


class Classification(object):
    def __init__(self):
        super(Classification, self).__init__()

    @staticmethod
    # todo
    def compmedDist(X: np.ndarray) -> np.float64:  # X:transpose of feature matrix
        size1 = X.shape[0]  # feature size
        Xmed = X

        # every value squared, and sum across row
        G = sum((Xmed * Xmed), 1)
        # expand (size1,) to (size1,size1), each row contains size1 sum
        Q = tile(G[:, newaxis], (1, size1))
        # R=Q.T
        R = tile(G, (size1, 1))

        dists = Q + R - 2 * dot(Xmed, Xmed.T)
        dists = dists - tril(dists)  # diagonal and right-top part set to 0
        dists = dists.reshape(size1 ** 2, 1, order='F').copy()
        return sqrt(0.5 * median(dists[dists > 0]))

    @staticmethod
    def kernel_Gaussian(x: ndarray, c: ndarray, sigma: np.float64) -> ndarray:
        (d, nx) = x.shape
        (d, nc) = c.shape
        x2 = sum(x ** 2, 0)
        c2 = sum(c ** 2, 0)

        distance2 = tile(c2, (nx, 1)) + tile(x2[:, newaxis], (1, nc)) - 2 * dot(x.T, c)

        return exp(-distance2 / (2 * (sigma ** 2)))

    @staticmethod
    def R_ULSIF(x_nu: ndarray, x_de: ndarray, alpha, sigma_list, lambda_list, b, fold):
        # x_nu: samples from numerator
        # x_de: samples from denominator
        # alpha: alpha defined in relative density ratio
        # sigma_list, lambda_list: parameters for model selection
        #
        # b: number of kernel basis
        # fold: number of fold for cross validation

        (d, n_nu) = x_nu.shape
        (d, n_de) = x_de.shape
        b = min(b, n_nu)
        x_ce = x_nu[:, np.r_[0:b]]

        score_cv = np.zeros((len(sigma_list), len(lambda_list)))
        cv_index_nu = np.random.permutation(n_nu)
        cv_split_nu = np.floor(np.r_[0:n_nu] * fold / n_nu)
        cv_index_de = np.random.permutation(n_de)
        cv_split_de = np.floor(np.r_[0:n_de] * fold / n_de)

        # 20 iterations
        iter = 1000
        count = 0
        mu = 0.1
        k1 = 0.1
        loss_old = float('inf')
        thetat = None
        for i in range(0, iter):
            count += 1
            print("current iteration:", count)
            # choose sigma
            for sigma_index in r_[0:size(sigma_list)]:  # try different sigma
                sigma = sigma_list[sigma_index]
                K_de = Classification.kernel_Gaussian(x_de, x_ce, sigma).T  # ndarray
                K_nu = Classification.kernel_Gaussian(x_nu, x_ce, sigma).T  # ndarray

                score_tmp = zeros((fold, size(lambda_list)))

                for k in np.r_[0:fold]:  # select k-fold
                    Ktmp1 = K_de[:, cv_index_de[cv_split_de != k]]
                    Ktmp2 = K_nu[:, cv_index_nu[cv_split_nu != k]]
                    # Ktmp:990,990
                    Ktmp = alpha / Ktmp2.shape[1] * fast_matmul(Ktmp2, Ktmp2.T) \
                           + (1 - alpha) / Ktmp1.shape[1] * np.dot(Ktmp1, Ktmp1.T)
                    # Ktmp = alpha / Ktmp2.shape[1] * np.dot(Ktmp2, Ktmp2.T) \
                    #        + (1 - alpha) / Ktmp1.shape[1] * np.dot(Ktmp1, Ktmp1.T)
                    # mKtmp:990,
                    mKtmp = np.mean(K_nu[:, cv_index_nu[cv_split_nu != k]], 1)

                    for lambda_index in np.r_[0:size(lambda_list)]:
                        lbd = lambda_list[lambda_index]
                        # thetat_cv:990,
                        thetat_cv = linalg.solve(Ktmp + (lbd * eye(b)), mKtmp)
                        thetah_cv = thetat_cv

                        score_tmp[k, lambda_index] = \
                            alpha * np.mean(
                                np.dot(K_nu[:, cv_index_nu[cv_split_nu == k]].T, thetah_cv) ** 2) / 2. \
                            + (1 - alpha) * np.mean(
                                np.dot(K_de[:, cv_index_de[cv_split_de == k]].T, thetah_cv) ** 2) / 2. \
                            - np.mean(dot(K_nu[:, cv_index_nu[cv_split_nu == k]].T, thetah_cv))

                    score_cv[sigma_index, :] = mean(score_tmp, 0)

            score_cv_tmp = score_cv.min(1)
            lambda_chosen_index = score_cv.argmin(1)

            score = score_cv_tmp.min()
            sigma_chosen_index = score_cv_tmp.argmin()

            lambda_chosen = lambda_list[lambda_chosen_index[sigma_chosen_index]]
            sigma_chosen = sigma_list[sigma_chosen_index]

            # compute coe
            K_de = Classification.kernel_Gaussian(x_de, x_ce, sigma_chosen).T
            K_nu = Classification.kernel_Gaussian(x_nu, x_ce, sigma_chosen).T

            coe = alpha * fast_matmul(K_nu, K_nu.T) / n_nu \
                  + (1 - alpha) * np.dot(K_de, K_de.T) / n_de \
                  + lambda_chosen * np.eye(b)
            var = np.mean(K_nu, 1)

            # solve theta
            thetat = linalg.solve(coe, var)
            thetatTranspose = thetat.transpose()

            # compute loss
            loss1 = dot(thetat.T, (alpha * dot(K_nu, K_nu.T) / n_nu - (1 - alpha) * dot(K_de, K_de.T) / n_de))
            loss_bias = 0.5 * dot(loss1, thetat) - dot(var.T, thetat) + 0.5 * lambda_chosen * dot(thetat.T, thetat)
            print("part 2 loss:", loss_bias)
            if alpha < 0:
                loss_alpha = -1
            elif alpha > 1:
                loss_alpha = 1
            else:
                loss_alpha = 0
            print("alpha loss:", loss_alpha)

            loss_new = loss_alpha + 1.0 * loss_bias

            # gradient
            result = dot(thetatTranspose, (dot(K_nu, K_nu.T) / n_nu - dot(K_de, K_de.T) / n_de))
            loss2change = 0.5 * dot(result, thetat)
            print("loss bias change:", loss2change)

            # update alpha
            alpha_old = alpha
            print("alpha old:", alpha_old)
            while True:
                mu = mu * exp(-k1 * i)
                if alpha < 0:
                    alpha = alpha - mu * (-1 + loss2change)
                if alpha > 1:
                    alpha = alpha - mu * (1 + loss2change)
                if 0 <= alpha <= 1:
                    alpha = alpha - mu * loss2change
                if alpha < 0 or alpha >= 1:
                    k1 = 2 * k1
                    alpha = alpha_old
                else:
                    k1 = 0.1
                    break
            print("mu:", mu)
            if abs(loss_old - loss_new) < 1e-7:
                break

            print("Old loss:", loss_old)
            print("new loss:", loss_new)
            print("alpha updated:", alpha)
            print("alpha change:", alpha_old - alpha)
            print("count", count)
            print()
            loss_old = loss_new

        # return result
        thetah = thetat
        wh_x_de = np.dot(K_de.T, thetah).T
        wh_x_nu = np.dot(K_nu.T, thetah).T
        print("wh_x_de:", wh_x_de)
        print("wh_x_de len:", len(wh_x_de))
        print("wh_x_nu:", wh_x_nu)
        print("wh_x_nu len:", len(wh_x_nu))

        wh_x_de[wh_x_de < 0] = 0

        return (thetah, wh_x_de, x_ce, sigma_chosen)

    @staticmethod
    def compute_target_weight(thetah, x_ce, sigma, x_nu):
        K_nu = Classification.kernel_Gaussian(x_nu, x_ce, sigma).T
        wh_x_nu = dot(K_nu.T, thetah).T
        wh_x_nu[wh_x_nu < 0.00000001] = 0.00000001
        return wh_x_nu

    @staticmethod
    def sigma_list(x_nu: np.array, x_de: np.array) -> ndarray:
        # concatenate tow ndarrays along columns
        x: np.array = c_[x_nu, x_de]
        med = Classification.compmedDist(x.T)
        return med * array([0.6, 0.8, 1, 1.2, 1.4])

    @staticmethod
    def lambda_list() -> ndarray:
        return array([10 ** (-3), 10 ** (-2), 10 ** (-1), 10 ** (0), 10 ** (1)])

    @staticmethod
    def read_csv(path: str, size=None, delimiter=",", header=None) -> Tuple[matrix, ndarray]:
        data: ndarray = None
        data_label: ndarray = None
        with open(path) as csvfile:
            count = 0
            if not header:
                reader = csv.reader(csvfile, delimiter=delimiter)
            else:
                reader = csv.DictReader(csvfile, fieldnames=header, delimiter=delimiter)
            for row in reader:
                tmp = [float(x) for x in row[:-1]]
                label = row[-1]
                if data is None:
                    data = np.array(tmp)
                else:
                    data = np.vstack((data, tmp))
                if data_label is None:
                    data_label = np.array(label)
                else:
                    data_label = np.vstack((data_label, label))
                count += 1
                if size and count > size:
                    break
            data: matrix = np.matrix(data, dtype=np.float64)
            return data, data_label

    @staticmethod
    def get_model(trgx_matrix: np.matrix,
                  srcx_matrix: np.matrix,
                  srcy_matrix: List[int],
                  alpha: float,
                  sigma_list: np.ndarray,
                  lambda_list: np.ndarray, b: int, fold, subsize):
        """
        :param m: a row matrix contains current parameter values corresponding to [Y, x1,x2,...,xn, 1]
        :param data_matrix: M*(N+1) matrix, M - number of instances, N - number of features, 
        the last column is target value
        :return: mu - MLE estimate value of y for each data point.
        """

        srcx_array = np.array(srcx_matrix.T)
        srcy_array = np.array(srcy_matrix)
        srcy_labelList = srcy_array
        subwindowSize = len(trgx_matrix) / subsize
        avgw = []
        for i in range(0, subsize):
            trgx_array = np.array(
                trgx_matrix[i * int(subwindowSize):(i + 1) * int(subwindowSize)].T)  # shihab: changed for python 3
            (thetah, w, x_ce, sigma) = Classification.R_ULSIF(trgx_array, srcx_array, alpha, sigma_list, lambda_list, b,
                                                              fold)
            avgw.append(w)
            break
        avg = np.average(avgw, axis=0)
        for i in range(0, len(avg)):
            with open('output/weight_norm15group1player.csv', 'a+') as f:
                writer = csv.writer(f)
                writer.writerow([avg[i]])
        params = {'gamma': [2 ** 2, 2 ** -16], 'C': [2 ** -6, 2 ** 15]}
        svr = svm.SVC()
        opt = GridSearchCV(svr, params)
        opt.fit(X=np.array(srcx_matrix), y=np.array(srcy_labelList))
        optParams = opt.best_params_
        # model = linear_model.RidgeClassifier(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None,
        #                                    tol=0.001, class_weight=None, solver='auto', random_state=None)

        model = svm.SVC(decision_function_shape='ovr', probability=True, C=optParams['C'], gamma=optParams['gamma'])
        # model1 = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
        # src_y = src_matrix[:, -1]

        # design a loss function to choosing the model in hypothesis set.
        print(srcx_array.shape, np.array(srcy_labelList).shape, len(avg))
        model.fit(np.array(srcx_matrix), np.array(srcy_labelList), avg)
        # model1.fit(srcx_matrix,src_y)
        # predictions = model.predict(np.array(trgx_matrix))
        # confidence = model.decision_function(np.array(trgx_matrix))
        # pred_lr = model1.predict(trgx_matrix)
        return model

    @staticmethod
    def get_predictlabel(trgx_matrix, model):

        predictions = model.predict(np.array(trgx_matrix))
        confidence = model.decision_function(np.array(trgx_matrix))
        # pred_lr = model1.predict(trgx_matrix)
        return predictions, confidence

    @staticmethod
    def get_true_label(target_label):
        y = target_label
        return y.transpose().tolist()[0]

    @staticmethod
    def get_prediction_error(prediction_label, true_label):
        error = 0.00
        for i in range(len(prediction_label)):
            if prediction_label[i] != true_label[i]:
                error = error + 1.00
                # error += abs(prediction_value[i]-true_value[i])
        return error / len(prediction_label)


if __name__ == '__main__':
    classification = Classification()
    # find execution time
    start_time = time.clock()

    srcx_matrix, srcy_matrix = Classification.read_csv(
        'datasets/syndata_002_normalized_no_novel_class_source_stream.csv', size=500)

    trgx_matrix, trgy_matrix = Classification.read_csv(
        'datasets/syndata_002_normalized_no_novel_class_target_stream.csv', size=500)

    alpha = 0.0
    b = trgx_matrix.T.shape[1]
    fold = 5
    sigma_list = Classification.sigma_list(np.array(trgx_matrix.T), np.array(srcx_matrix.T))
    lambda_list = Classification.lambda_list()
    # srcx_matrix = matrix_[:, :-1]
    # trgx_matrix = target_[:, :-1]
    srcx_array = np.array(srcx_matrix.T)
    trgx_array = np.array(trgx_matrix.T)
    # (PE, w, s) = Classification.R_ULSIF(trgx_array, srcx_array, alpha, sigma_list, lambda_list, b, fold)
    # w_de = w[0:srcx_array.shape[1]]
    model = Classification.get_model(trgx_matrix, srcx_matrix, srcy_matrix, alpha, sigma_list, lambda_list, b, fold, 2)
    predict_label, confidence = Classification.get_predictlabel(trgx_matrix, model)
    true_label = Classification.get_true_label(trgy_matrix)
    truetrg_labellist = []
    for i in range(len(np.array(trgy_matrix))):
        if np.array(trgy_matrix)[i] == 'class1':
            truetrg_labellist.append(1)
        if np.array(trgy_matrix)[i] == 'class2':
            truetrg_labellist.append(2)
        if np.array(trgy_matrix)[i] == 'class3':
            truetrg_labellist.append(3)
        if np.array(trgy_matrix)[i] == 'class4':
            truetrg_labellist.append(4)
        if np.array(trgy_matrix)[i] == 'class5':
            truetrg_labellist.append(5)
        if np.array(trgy_matrix)[i] == 'class6':
            truetrg_labellist.append(6)
        if np.array(trgy_matrix)[i] == 'class7':
            truetrg_labellist.append(7)
    print("true_label", truetrg_labellist)
    print("len true_label", len(truetrg_labellist))
    print("predict_label", predict_label)
    print("len predict_label", len(predict_label))
    err = Classification.get_prediction_error(predict_label, truetrg_labellist)

    print("the err:", err)
