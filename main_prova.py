# main_prova.py

import numpy as np
from solver.jacobi import jacobi_method  # importa la funzione dal modulo nella cartella solver


def main():
    # === Creazione di una matrice A e dei vettori b e x0 ===
    A = np.array([[4.0, -1.0, 0.0],
                  [-1.0, 4.0, -1.0],
                  [0.0, -1.0, 3.0]])

    b = np.array([15.0, 10.0, 10.0])  # vettore dei termini noti
    x0 = np.zeros_like(b)  # vettore iniziale (tutti zeri)

    # === Chiamata al metodo di Jacobi ===
    x_sol, nit, tempo, err = jacobi_method(A, b, x0, tol=1e-4, max_iter=20000)

    # === Output dei risultati ===
    print("✅ Soluzione trovata:", x_sol)
    print("🌀 Iterazioni effettuate:", nit)
    print("⏱️ Tempo impiegato:", tempo, "secondi")
    print("📉 Errore finale:", err)


if __name__ == "__main__":
    main()
