from scipy.io import mmread
import os

# VERIFICA ESISTENZA FILE
count = 0
# spa1
spa1_path = "C:\\Users\\Samuele Roncoroni\\Desktop\\Matrici\\spa1.mtx"
if os.path.isfile(spa1_path) and spa1_path.endswith("spa1.mtx"):
    print("✅ Percorso corretto! File spa1.mtx trovato.")

    # Carica la matrice
    matrix = mmread(spa1_path)
    print("✅ Matrice caricata correttamente.\n")
    count += 1
else:
    print("❌ Il percorso non è valido o non è un file .mtx.\n")

# spa2
spa2_path = "C:\\Users\\Samuele Roncoroni\\Desktop\\Matrici\\spa2.mtx"
if os.path.isfile(spa2_path) and spa2_path.endswith("spa2.mtx"):
    print("✅ Percorso corretto! File spa2.mtx trovato.")

    # Carica la matrice
    matrix = mmread(spa2_path)
    print("✅ Matrice caricata correttamente.\n")
    count += 1
else:
    print("❌ Il percorso non è valido o non è un file .mtx.")

# vem1
vem1_path = "C:\\Users\\Samuele Roncoroni\\Desktop\\Matrici\\vem1.mtx"
if os.path.isfile(vem1_path) and vem1_path.endswith("vem1.mtx"):
    print("✅ Percorso corretto! File vem1.mtx trovato.")

    # Carica la matrice
    matrix = mmread(vem1_path)
    print("✅ Matrice caricata correttamente.\n")
    count += 1
else:
    print("❌ Il percorso non è valido o non è un file .mtx.\n")

# vem2
vem2_path = "C:\\Users\\Samuele Roncoroni\\Desktop\\Matrici\\vem2.mtx"
if os.path.isfile(vem2_path) and vem2_path.endswith("vem2.mtx"):
    print("✅ Percorso corretto! File vem2.mtx trovato.")

    # Carica la matrice
    matrix = mmread(vem2_path)
    print("✅ Matrice caricata correttamente.\n")
    count += 1
else:
    print("❌ Il percorso non è valido o non è un file .mtx.\n")

if count == 4:
    print("✅Tutte le matrici sono state caricate correttamente")
else:
    print("❌Non tutte le matrici sono state caricate correttamente")

# Aggiungere controllo nel caso di nuova matrice: se non ha l'intestazione esatta va aggiunta
