from scipy import sparse
import numpy as np


# store sparse matrix in memory
eye = np.eye(4)
sparse_matrix = sparse.csr_matrix(eye)
print("\n Scipy sparse CSR Matrix:\n %s" % sparse_matrix)