import numpy as np
from scipy.sparse import csr_matrix


def lower_triangular_method(L, b):
    """
    Risoluzione esplicita del sistema triangolare inferiore Lx = b.

    Parametri:
    - L: matrice triangolare inferiore (densa o sparsa convertita in densa)
    - b: vettore dei termini noti

    Restituisce:
    - x: soluzione del sistema
    """
    L = L.toarray() if isinstance(L, csr_matrix) else L
    n = L.shape[0]
    x = np.zeros(n)

    for i in range(n):
        sum_Lx = np.dot(L[i, :i], x[:i])
        x[i] = (b[i] - sum_Lx) / L[i, i]

    return x
