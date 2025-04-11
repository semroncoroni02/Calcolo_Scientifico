import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix
from solver.jacobi import jacobi_method
from solver.gauss_seidel import gauss_seidel_method
from solver.gradiente import gradient_method
from solver.gradiente_coniugato import conjugate_gradient_method
from utils.checks import validate_matrix
import time


def main():
    # === 1. Richiede il path della matrice da analizzare ===
    path = input("\nInserire il path della matrice da analizzare --> ")
    matrix_path = path

    print(f"\nCaricamento matrice da file: {matrix_path} ...")

    try:
        # === 2. Lettura e conversione ===
        A = mmread(matrix_path)
        A = csr_matrix(A)

        print(f"Matrice caricata con dimensioni: {A.shape[0]} x {A.shape[1]}")

        # === 3. Vettori noti ed esatti ===
        x_exact = np.ones(A.shape[1])
        b = A @ x_exact

        # === 4. Validazione ===
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

    # === 5. Inizializzazione ===
    print("\n✔️ Tutti i controlli superati. È ora possibile applicare i metodi iterativi.\n")

    # === METODI ITERATIVI ===
    print("\n\033[33m♻ METODI ITERATIVI ♻\033[0m")
    x0 = np.zeros_like(b)
    # === Scelta della tolleranza da parte dell'utente ===
    tolleranze = [1e-4, 1e-6, 1e-8, 1e-10]
    print("\n📌 Scegli la tolleranza desiderata per i metodi:")
    for i, tol_val in enumerate(tolleranze, start=1):
        print(f"  {i}. {tol_val:.0e}")

    scelta = input("Inserire il numero corrispondente alla tolleranza [1-4] --> ")

    try:
        idx = int(scelta.strip()) - 1
        if idx not in range(len(tolleranze)):
            raise ValueError
        tol = tolleranze[idx]
        print(f"\n👉 Tolleranza selezionata: {tol:.0e}")
    except ValueError:
        print("⚠️ Scelta non valida. Verrà usata la tolleranza di default: 1e-10.")
        tol = 1e-10

    max_iter = 20000

    # === METODO DI JACOBI ===
    print("\n📌 Metodo di Jacobi...")
    x_jacobi, nit_j, time_j, err_j, rel_err_j = jacobi_method(A, b, x0, x_exact, tol, max_iter)

    # === METODO DI GAUSS-SEIDEL ===
    print("📌 Metodo di Gauss-Seidel...")
    x_gs, nit_gs, time_gs, err_gs, rel_err_gs = gauss_seidel_method(A, b, x0, x_exact, tol, max_iter)

    # === METODO DEL GRADIENTE ===
    print("📌 Metodo del Gradiente...")
    x_grad, nit_grad, time_grad, err_grad, rel_err_grad = gradient_method(A, b, x0, x_exact, tol, max_iter)

    # === METODO DEL GRADIENTE CONIUGATO ===
    print("📌 Metodo del Gradiente Coniugato...\n")
    x_cg, nit_cg, time_cg, err_cg, rel_err_cg = conjugate_gradient_method(A, b, x0, x_exact, tol, max_iter)

    # === RISULTATI AGGREGATI ===
    print("- " * 45)
    print(f"📊 \033[32mRISULTATI FINALI\033[0m 📊  | 🎲 \033[32mMATRICE --> [{path[-8:]}]\033[0m 🎲 |  ⚠️ \033[32mTOLLERANZA --> [{tol:.0e}]\033[0m ⚠️")
    #print(f"🎲 MATRICE --> [{path[-8:]}] 🎲")
    #print(f"⚠️ TOLLERANZA --> [{tol:.0e}] ⚠️")
    print("-" * 89)
    # print(f"\n👉 Tolleranza selezionata: {tol:.0e}")
    print("Metodo                 | 🌀Iterazioni | ⏱️Tempo (s) | 📉Errore Finale | 📏Errore Relativo")
    print("-" * 89)
    print(f"Jacobi                 | {nit_j:10d}   | {time_j:9.4f}   | {err_j:13.2e}   | {rel_err_j:16.2e}")
    print(f"Gauss-Seidel           | {nit_gs:10d}   | {time_gs:9.4f}   | {err_gs:13.2e}   | {rel_err_gs:16.2e}")
    print(f"Gradiente              | {nit_grad:10d}   | {time_grad:9.4f}   | {err_grad:13.2e}   | {rel_err_grad:16.2e}")
    print(f"Gradiente Coniugato    | {nit_cg:10d}   | {time_cg:9.4f}   | {err_cg:13.2e}   | {rel_err_cg:16.2e}")
    print("- " * 45)

# Punto di ingresso principale
if __name__ == "__main__":
    main()
