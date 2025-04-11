import numpy as np
import time
from scipy.sparse import csr_matrix


def gradient_method(A, b, x0, x_exact, tol, max_iter):
    """
    Metodo del Gradiente (Steepest Descent) per la risoluzione di Ax = b.

    Parametri:
    - A : scipy.sparse.csr_matrix
        Matrice dei coefficienti (simmetrica definita positiva)
    - b : np.ndarray
        Vettore dei termini noti
    - x0 : np.ndarray
        Vettore iniziale (guess)
    - x_exact : np.ndarray
        Soluzione esatta per il calcolo dell'errore relativo
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
        Errore (relativo al residuo, norm(b - A x_new)/norm(x_new))
    - relative_error : float
        Errore relativo rispetto alla soluzione esatta
    """
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)

    x_old = x0.copy()
    nit = 0
    err = 1.0
    start_time = time.time()

    while nit < max_iter and err > tol:
        r = b - A @ x_old  # residuo
        Ar = A @ r
        # Passo ottimale: alpha = (r^T r) / (r^T (A r))
        alpha = (r @ r) / (r @ Ar)
        x_new = x_old + alpha * r

        # Errore calcolato come il residuo relativo
        err = np.linalg.norm(b - A @ x_new) / np.linalg.norm(x_new)
        x_old = x_new.copy()
        nit += 1

    elapsed_time = time.time() - start_time
    relative_error = np.linalg.norm(x_new - x_exact, ord=np.inf) / np.linalg.norm(x_exact, ord=np.inf)

    if nit == max_iter and err > tol:
        print("⚠️ Metodo del Gradiente: raggiunto il numero massimo di iterazioni senza convergenza.")

    return x_new, nit, elapsed_time, err, relative_error
