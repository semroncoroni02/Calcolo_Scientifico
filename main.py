import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix
from solver.jacobi import jacobi_method
from solver.gauss_seidel import gauss_seidel_method
from solver.gradiente import gradient_method
from solver.gradiente_coniugato import conjugate_gradient_method
from utils.checks import validate_matrix
import matplotlib.pyplot as plt


def main():
    path = input("\nInserire il path della matrice da analizzare --> ")
    matrix_path = path

    print(f"\nCaricamento matrice da file: {matrix_path} ...")

    try:
        A = mmread(matrix_path)
        A = csr_matrix(A)
        print(f"Matrice caricata con dimensioni: {A.shape[0]} x {A.shape[1]}")

        x_exact = np.ones(A.shape[1])
        b = A @ x_exact

        print("\nAvvio dei controlli sulla matrice...\n")
        valid, message = validate_matrix(A, b)

        if not valid:
            print("❌ MATRICE NON VALIDA:", message)
            return
        print("✅ MATRICE VALIDA:", message)

    except FileNotFoundError:
        print(f"Errore: il file '{matrix_path}' non è stato trovato.")
        return
    except Exception as e:
        print(f"\033[33mErrore durante il caricamento o la validazione della matrice: {e}\033[0m")
        return

    print("✔️ Tutti i controlli superati. È ora possibile applicare i metodi iterativi.\n")

    print("\033[33m♻ ♻ ♻ METODI ITERATIVI ♻ ♻ ♻\033[0m")
    x0 = np.zeros_like(b)
    tolleranze = [1e-4, 1e-6, 1e-8, 1e-10]
    max_iter = 20000

    # Per ogni tolleranza applica tutti i metodi e salva i risultati
    risultati = {tol: {} for tol in tolleranze}

    for tol in tolleranze:
        print(f"\n👉 Tolleranza in esame: {tol:.0e}")

        risultati[tol]['Jacobi'] = jacobi_method(A, b, x0, x_exact, tol, max_iter)
        risultati[tol]['Gauss-Seidel'] = gauss_seidel_method(A, b, x0, x_exact, tol, max_iter)
        risultati[tol]['Gradiente'] = gradient_method(A, b, x0, x_exact, tol, max_iter)
        risultati[tol]['Gradiente Coniugato'] = conjugate_gradient_method(A, b, x0, x_exact, tol, max_iter)

        print("- " * 46)
        print(
            f"📊 \033[32mRISULTATI FINALI\033[0m 📊  | 🎲 \033[32mMATRICE --> [{path[-8:]}]\033[0m 🎲 |  ⚠️ \033[32mTOLLERANZA --> [{tol:.0e}]\033[0m ⚠️")
        print("-" * 91)
        print("Metodo                  | 🌀Iterazioni | ⏱️Tempo (s) | 📉Errore Finale | 📏Errore Relativo")
        print("-" * 91)
        for metodo, (x_sol, nit, tempo, err, rel_err) in risultati[tol].items():
            print(f"{metodo:<23} | {nit:10d}   | {tempo:9.4f}   | {err:13.2e}   | {rel_err:16.2e}")
        print("- " * 46)

    # === GRAFICI COMPARATIVI ===
    metodi = ['Jacobi', 'Gauss-Seidel', 'Gradiente', 'Gradiente Coniugato']
    colori = ['b', 'g', 'orange', 'r']

    # Iterazioni
    plt.figure(figsize=(10, 6))
    for metodo, colore in zip(metodi, colori):
        valori = [risultati[tol][metodo][1] for tol in tolleranze]
        plt.plot(tolleranze, valori, 'o-', label=metodo, color=colore)
    plt.xscale('log')
    plt.xlabel('Tolleranza')
    plt.ylabel('Numero di Iterazioni')
    plt.title('Confronto Iterazioni vs Tolleranza')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Tempo
    plt.figure(figsize=(10, 6))
    for metodo, colore in zip(metodi, colori):
        valori = [risultati[tol][metodo][2] for tol in tolleranze]
        plt.plot(tolleranze, valori, 'o-', label=metodo, color=colore)
    plt.xscale('log')
    plt.xlabel('Tolleranza')
    plt.ylabel('Tempo di Esecuzione (s)')
    plt.title('Confronto Tempo vs Tolleranza')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Errore Relativo
    plt.figure(figsize=(10, 6))
    for metodo, colore in zip(metodi, colori):
        valori = [risultati[tol][metodo][4] for tol in tolleranze]
        plt.plot(tolleranze, valori, 'o-', label=metodo, color=colore)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Tolleranza')
    plt.ylabel('Errore Relativo')
    plt.title('Confronto Errore Relativo vs Tolleranza')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
