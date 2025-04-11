import numpy as np
import time
from scipy.sparse import csr_matrix


def jacobi_method(A, b, x0, x_exact, tol, max_iter):
    """
    Metodo di Jacobi per la risoluzione di Ax = b con supporto per matrici sparse.

    Parametri:
    - A : scipy.sparse.csr_matrix
        Matrice dei coefficienti (quadrata, con diagonale non nulla)
    - b : np.ndarray
        Vettore dei termini noti
    - x0 : np.ndarray
        Vettore iniziale
    - x_exact : np.ndarray
        Vettore della soluzione esatta (usato per il calcolo dell'errore relativo)
    - tol : float
        Tolleranza per il criterio di arresto
    - max_iter : int
        Numero massimo di iterazioni

    Restituisce:
    - x_new : np.ndarray
        Soluzione approssimata
    - nit : int
        Numero di iterazioni effettuate
    - elapsed_time : float
        Tempo impiegato in secondi
    - err : float
        Errore (norma infinito tra due iterazioni successive)
    - relative_error : float
        Errore relativo rispetto alla soluzione esatta
    """
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)

    # Estrai la diagonale e calcola il suo inverso (element-wise)
    D = A.diagonal()
    D_inv = 1.0 / D

    x_old = x0.copy()
    x_new = np.zeros_like(x0)
    start_time = time.time()

    for nit in range(1, max_iter + 1):
        # Aggiornamento simultaneo:
        r = b - A @ x_old
        x_new = x_old + D_inv * r

        err = np.linalg.norm(x_new - x_old, ord=np.inf)
        if err < tol:
            break
        x_old = x_new.copy()

    elapsed_time = time.time() - start_time

    if nit == max_iter and err >= tol:
        print("⚠️ Metodo Jacobi: raggiunto il numero massimo di iterazioni senza convergenza.")

    relative_error = np.linalg.norm(x_new - x_exact, ord=np.inf) / np.linalg.norm(x_exact, ord=np.inf)
    return x_new, nit, elapsed_time, err, relative_error
