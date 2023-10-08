from PositionalIndex import PositionalIndex
# from sklearn.linear_model import LinearRegression
import numpy as np
from Utils import *
import math
from scipy import stats

if __name__ == "__main__":
    positionalIndex = PositionalIndex()
    # positionalIndex.startGettingQueries()
    positionalIndex.resolveQueriesWithVectorSpace()
    # positionalIndex.resolveQueryWithWordEmbedding()

    # b_0 = 1.1005761522193787
    # b_1 = 0.5442159467570922
    # print(len(positionalIndex.positionalIndexDict))
    # print(positionalIndex.numberOfTokens)
    # M = pow(10, b_0 + b_1 * math.log(positionalIndex.numberOfTokens, 10))
    # print(M)

    # vocabulariSizesList = []
    # collectionSizesList = []

    # for i in range(1, 5):
    #   rowCounts = 500 * i
    #   positionalIndex.getXlsxData(rowCounts)
    #   positionalIndex.getPositionalIndex()
    #   print(len(positionalIndex.positionalIndexDict))
    #   print(positionalIndex.numberOfTokens)

    # y = np.array([8516, 11865, 14193, 19537])
    # y_prim = np.log10(y)
    # x = np.array([151702, 299162, 450051, 665027])
    # x_prim = np.log10(x)
    #
    # b = estimate_coef(x_prim, y_prim)
    # print("Estimated coefficients:\nb_0 = {}  \
    #           \nb_1 = {}".format(b[0], b[1]))
    # plot_regression_line(x_prim, y_prim, b)
