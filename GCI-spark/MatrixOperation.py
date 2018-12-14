import time

from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg import DenseMatrix
from pyspark.sql import SparkSession
from pyspark.mllib.linalg.distributed import BlockMatrix, IndexedRow, IndexedRowMatrix, RowMatrix
import numpy as np

conf = SparkConf().setMaster("local").setAppName("sample")
sc = SparkContext(conf=conf)
sparkSession = SparkSession(sc)

counter = 1

# use BlockMatrix, too slow
def multiply_transpose2(A: np.array) -> np.ndarray:  # A*A.T
    global counter
    print()
    print("No." + str(counter) + " matrix multiplication starts")
    start_time = time.time()
    print("matrix shape:", A.shape)
    listA = A.tolist()
    rddA = sc.parallelize([IndexedRow(i, listA[i]) for i in range(len(listA))])
    matA = IndexedRowMatrix(rddA).toBlockMatrix()
    matT = matA.transpose()
    matR = matA.multiply(matT)
    res = matR.toLocalMatrix().toArray()
    elapsed_time = time.time() - start_time
    print("No." + str(counter) + " matrix multiplication ends, takes time:", elapsed_time)
    counter = counter + 1
    return res

# use RowMatrix, 4 times faster than BlockMatrix
def multiply_transpose(A: np.array) -> np.ndarray:  # A*A.T
    return multiply_matrices(A, A.T)

# use BlockMatrix, too slow
def multiply_matrices2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    listA = A.tolist()
    rddA = sc.parallelize([IndexedRow(i, listA[i]) for i in range(len(listA))])
    matA = IndexedRowMatrix(rddA).toBlockMatrix()

    listB = B.tolist()
    rddB = sc.parallelize([IndexedRow(i, listB[i]) for i in range(len(listB))])
    matB = IndexedRowMatrix(rddB).toBlockMatrix()

    matC = matA.multiply(matB).toLocalMatrix()
    return matC.toArray()

# use RowMatrix, 4 times faster than BlockMatrix
def multiply_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    global counter
    print()
    print("No." + str(counter) + " matrix multiplication starts")
    start_time = time.time()
    print("matrix shape:", A.shape)
    rddA = sc.parallelize(A.tolist())
    matA = RowMatrix(rddA)

    matB = DenseMatrix(B.shape[0], B.shape[1], B.flatten().tolist(), isTransposed=True)

    matC = matA.multiply(matB)
    rows = matC.rows.collect()
    res = np.array([row.toArray() for row in rows])
    elapsed_time = time.time() - start_time
    print("No." + str(counter) + " matrix multiplication ends, takes time:", elapsed_time)
    counter = counter + 1
    return res


if __name__ == '__main__':
    A = np.array([[1, 2, 3],
                  [3, 4, 5],
                  [5, 6, 7],
                  [7, 8, 9]], dtype=np.float64)
    print(A.shape)

    C = multiply_matrices(A, A.T)
    print(C)
    print()

    C = multiply_transpose(A)
    print(C)
    print()

    C = np.matmul(A, A.T)
    print(C)
    print()
