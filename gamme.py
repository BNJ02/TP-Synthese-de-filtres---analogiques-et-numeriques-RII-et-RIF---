import numpy as np

def gamme(duree: float, fe: float) -> np.ndarray:
    """
    Génère une gamme de notes à partir des fréquences fondamentales
    correspondant à la gamme chromatique (do, do#, ré, ré#, mi, fa, fa#,
    sol, sol#, la, la#, si)

    Paramètres :
        duree (float) : durée totale de la gamme en secondes
        fe (float) : fréquence d'échantillonnage en Hz

    Retourne :
        gam (numpy.ndarray) : tableau 1D contenant les échantillons de la
                                gamme de notes
        t (numpy.ndarray) : tableau 1D contenant les échantillons de temps
                             correspondants, allant de 0 à duree avec un
                             pas de 1/fe
    """
    # Fréquences fondamentales des notes de la gamme chromatique
    freq_notes = np.array([262, 294, 330, 349, 392, 440, 494, 523])

    # Génère un tableau d'échantillons de temps allant de 0 à duree avec un
    # pas de 1/fe
    t = np.arange(0, duree + 1/fe, 1/fe)

    # Crée un tableau de zéros de taille (len(t), len(freq_notes))), avec une
    # colonne pour chaque note et une ligne pour chaque échantillon
    gam = np.zeros(len(t))

    # Durée de chaque note
    duree_note = duree / len(freq_notes)

    # Pour chaque note dans la gamme, génère un signal sinusoïdal
    # correspondant à sa fréquence fondamentale et l'ajoute à la colonne
    # appropriée de gam
    for i in range(len(freq_notes)):
        # Génère un tableau d'échantillons de temps pour la note actuelle
        t_note = np.arange(i*duree_note, (i+1)*duree_note, 1/fe)
        # Génère le signal sinusoïdal pour la note actuelle
        signal_note = np.sin(2 * np.pi * freq_notes[i] * t_note)
        # Ajoute le signal de la note actuelle à gam aux bons indices
        gam[int(i*duree_note*fe):int((i+1)*duree_note*fe)] = signal_note

    # Aplatit le tableau gam en un tableau 1D
    gam = np.ravel(gam)

    # Retourne le tableau d'échantillons de la gamme et le tableau
    # d'échantillons de temps correspondants
    return gam, t
