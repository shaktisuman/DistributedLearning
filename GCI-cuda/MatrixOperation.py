
from numba import cuda, guvectorize, void, float64
import numpy as np


@guvectorize([void(float64[:,:], float64[:,:], float64[:,:])], '(m,l),(l,n)->(m,n)', target='cuda')
def matmul_gu3(A, B, out):
    """Perform square matrix multiplication of out = A * B
    """
    # out = np.empty((A.shape[0], B.shape[1]), A.dtype)
    i, j = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        # print(tmp)
        out[i, j] = tmp

TPB=32
@cuda.jit
def fast_matmul_(A, B, C):
    # C = np.empty((A.shape[0], B.shape[1]), A.dtype)
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float64)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float64)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

def fast_matmul(A, B):
    C = np.empty((A.shape[0], B.shape[1]), dtype=np.float64)
    fast_matmul_(A, B, C)
    return C

if __name__ == '__main__':
    A = np.array([[1, 2, 3],
                  [3, 4, 5],
                  [5, 8, 7],
                  [7, 8, 9]], dtype=np.float64)
    print(A.shape)

    C = np.matmul(A, A.T)
    print(C)
    print()

    C = fast_matmul(A, A.T)
    print(C)
    print()