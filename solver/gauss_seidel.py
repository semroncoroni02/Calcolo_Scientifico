import numpy as np
import time
from scipy.sparse import csr_matrix, tril
from solver.lower_triangular import lower_triangular_method


def gauss_seidel_method(A, b, x0, x_exact, tol, max_iter):
    """
    Metodo di Gauss-Seidel per la risoluzione di Ax = b con supporto per matrici sparse.

    :parameter:
    - A : scipy.sparse.csr_matrix
        Matrice dei coefficienti (assunta quadrata e con diagonale non nulla)
    - b : np.ndarray
        Vettore dei termini noti
    - x0 : np.ndarray
        Vettore iniziale (guess)
    - x_exact : np.ndarray
        Soluzione esatta per il calcolo dell'errore relativo
    - tol : float
        Tolleranza per il criterio di arresto (norma infinito tra iterazioni)
    - max_iter : int
        Numero massimo di iterazioni

    :return:
    - x : np.ndarray
        Soluzione approssimata
    - nit : int
        Numero di iterazioni
    - elapsed_time : float
        Tempo impiegato in secondi
    - err : float
        Errore finale (norma inf tra due iterazioni)
    - relative_error : float
        Errore relativo rispetto alla soluzione esatta
    """
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)

        # Usa la funzione 'tril' per estrarre la parte inferiore in formato sparse
    L = tril(A, format="csr")
    U = A - L  # Parte strettamente superiore

    x_old = x0.copy()
    nit = 0
    err = np.inf
    start_time = time.time()

    while err > tol and nit < max_iter:
        rhs = b - U @ x_old
        # Risolvi il sistema triangolare inferiore
        x_new = lower_triangular_method(L, rhs)
        err = np.linalg.norm(x_new - x_old, ord=np.inf)
        x_old = x_new.copy()
        nit += 1

    elapsed_time = time.time() - start_time
    relative_error = np.linalg.norm(x_new - x_exact, ord=np.inf) / np.linalg.norm(x_exact, ord=np.inf)

    if nit == max_iter and err >= tol:
        print("⚠️ Metodo Gauss-Seidel: raggiunto il numero massimo di iterazioni senza convergenza.")

    return x_new, nit, elapsed_time, err, relative_error
