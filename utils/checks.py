import numpy as np

"""
Modulo checks.py

Contiene funzioni di controllo per verificare che una matrice soddisfi i requisiti necessari
per l'applicazione dei metodi iterativi:
  - simmetrica
  - definita positiva
  - compatibilità dimensionale
  - ...

Questi controlli sono fondamentali perché:
- I metodi del Gradiente e del Gradiente Coniugato richiedono matrici simmetriche e definite positive
- I metodi iterativi in generale richiedono matrici quadrate e compatibili con il vettore b
"""


def is_square(A):
    """
    Verifica che la matrice A sia quadrata (necessario per la risoluzione di sistemi lineari).
    In generale, data una matrice M[righe,colonne] --> M.shape[0] = n°righe & M.shape[1] = n°colonne

    :param A:matrice
    :return: True se A è quadrata, False altrimenti
    """
    return A.shape[0] == A.shape[1]


def is_nonzerodiagonal(A):
    """
    Verifica che la matrice A non contenga zeri sulla diagonale.

    :param A: matrice sparsa
    :return: True se tutti gli elementi diagonali sono diversi da zero, False altrimenti
    """
    return np.all(A.diagonal() != 0)


def is_nonzero(A):
    """
    Verifica che la matrice A non sia una matrice nulla (ovvero che contenga almeno un elemento diverso da 0).

    :param A: matrice sparsa

    :return: True se la matrice ha almeno un elemento non nullo, False altrimenti
    """
    return A.nnz > 0  # .nnz restituisce il numero di elementi diversi da 0


def is_symmetric(A, tol=1e-10):
    """
    Verifica che la matrice A sia simmetrica, ovvero A == A.T (trasposta).

    Per evitare problemi di arrotondamento numerico, si usa np.allclose con una tolleranza.

    :parameter:
    - A: matrice sparsa
    - tol: tolleranza numerica per il confronto (default 1e-10)

    :return: True se la matrice è simmetrica entro la tolleranza, False altrimenti
    """
    return np.allclose(A.toarray(), A.T.toarray(), atol=tol)


# A.toarray(): converte la matrice sparsa A in una matrice densa NumPy.
# A.T.toarray(): fa la trasposta di A, poi la converte in array.
# np.allclose(...): confronta elemento per elemento le due matrici, e ritorna True se tutti gli elementi sono "vicini" entro la tolleranza tol.
# tol=1e-10 significa: considera "uguali" due numeri se differiscono meno di 10^-10, per gestire piccole imprecisioni numeriche.


def is_positive_definite(A):
    """
    Verifica che la matrice A sia definita positiva.
    Una matrice è definita positiva se tutti i suoi autovalori sono strettamente positivi.

    ATTENZIONE: Questa verifica comporta il calcolo degli autovalori, che può essere costoso per matrici di grandi dimensioni.

    :param A: matrice sparsa

    :return: True se tutti gli autovalori sono positivi, False altrimenti
    """
    try:
        # Utilizziamo eigvalsh poiché A è (o dovrebbe essere) simmetrica
        eigvals = np.linalg.eigvalsh(A.toarray())
        return np.all(eigvals > 0)
    except:
        # Se il calcolo degli autovalori fallisce, consideriamo la matrice non valida
        return False


# A.toarray(): converte la matrice in array.
# np.linalg.eigvalsh(...): calcola gli autovalori reali di una matrice simmetrica in modo efficiente.
# np.all(eigvals > 0): controlla se tutti gli autovalori sono > 0.
# Se sì, la matrice è definita positiva

def is_compatible(A, b):
    """
    Verifica che la dimensione del vettore b sia compatibile con la matrice A
    (cioè che il numero di righe di A corrisponda alla dimensione di b).

    :parameter:
    - A: matrice
    - b: vettore

    :return: True se le dimensioni sono compatibili, False altrimenti
    """
    return A.shape[0] == b.shape[0]


def validate_matrix(A, b):
    """
    Funzione generale che esegue tutti i controlli fondamentali su A e b.

    :parameter:
    - A: matrice (idealmente simmetrica e definita positiva)
    - b: vettore dei termini noti del sistema

    :return:
    - (True, messaggio) se tutti i controlli sono superati
    - (False, messaggio di errore) se almeno un controllo fallisce
    """
    if not is_square(A):
        return False, "La matrice non è quadrata."
    if not is_nonzerodiagonal(A):
        return False, "La matrice presenta degli zeri sulla diagonale"
    if not is_nonzero(A):
        return False, "La matrice è nulla (tutti zeri)."
    if not is_symmetric(A):
        return False, "La matrice non è simmetrica."
    if not is_positive_definite(A):
        return False, "La matrice non è definita positiva."
    if not is_compatible(A, b):
        return False, "Il vettore b non è compatibile con la matrice A."

    return True, "La matrice è valida per l'applicazione dei metodi iterativi."
