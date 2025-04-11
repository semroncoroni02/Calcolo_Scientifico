import numpy as np
import time
from scipy.sparse import csr_matrix


def conjugate_gradient_method(A, b, x0, x_exact, tol, max_iter):
    """
    Metodo del Gradiente Coniugato per la risoluzione di Ax = b.

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
        Errore relativo finale (ad esempio, residuo relativo)
    - relative_error : float
        Errore relativo rispetto alla soluzione esatta
    """
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)

    x_old = x0.copy()
    r_old = b - A @ x_old
    p_old = r_old.copy()
    nit = 0
    # Utilizzo della norma del residuo iniziale come errore
    err = np.linalg.norm(r_old) / np.linalg.norm(b)
    start_time = time.time()

    while nit < max_iter and err > tol:
        Ap = A @ p_old
        alpha = (r_old @ r_old) / (p_old @ Ap)
        x_new = x_old + alpha * p_old
        r_new = r_old - alpha * Ap

        # Controllo sul residuo relativo
        err = np.linalg.norm(r_new) / np.linalg.norm(b)
        if err < tol:
            x_old = x_new.copy()
            break

        beta = (r_new @ r_new) / (r_old @ r_old)
        p_new = r_new + beta * p_old

        x_old = x_new.copy()
        r_old = r_new.copy()
        p_old = p_new.copy()
        nit += 1

    elapsed_time = time.time() - start_time
    relative_error = np.linalg.norm(x_new - x_exact, ord=np.inf) / np.linalg.norm(x_exact, ord=np.inf)

    if nit == max_iter and err > tol:
        print("⚠️ Metodo del Gradiente Coniugato: raggiunto il numero massimo di iterazioni senza convergenza.")

    return x_new, nit, elapsed_time, err, relative_error
