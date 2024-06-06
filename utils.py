import matplotlib.pyplot as plt
import numpy as np


def plot_filtered_signal(t: np.ndarray, sig_filtre: np.ndarray, f: np.ndarray, H: np.ndarray, w_norm: float, fe: float, fmin: float, fmax: float, Ap: float, Aa: float, fc: list, df: int, title: str) -> None:
    """
    Plot the filtered signal in time domain and the frequency response of the filter in amplitude in dB.
    :param t: time vector
    :param sig_filtre: filtered signal
    :param f: frequency vector
    :param H: frequency response
    :param w_norm: normalized angular frequency
    :param fe: sampling frequency
    :param fmin: minimum frequency
    :param fmax: maximum frequency
    :param Ap: passband ripple
    :param Aa: stopband attenuation
    :param fc: cutoff frequency
    :param df: transition bandwidth
    :param title: title of the plot    
    """
    # Nom de la window pour les tracés
    plt.figure('Filtre ' + title)

    # Tracé de le signal temporel filtré avec le filtre de Chebychev 2
    plt.subplot(2, 1, 1)
    plt.title('Signal temporel filtré - ' + title)
    plt.plot(t, sig_filtre, label='Signal filtré')
    plt.xlabel('Temps (s)')
    plt.ylabel('Amplitude (V)')
    plt.grid()

    # Tracé de la réponse fréquentielle du filtre en amplitude en dB
    S_filtre = np.fft.fft(sig_filtre) / fe
    f = np.fft.fftfreq(sig_filtre.shape[0], d=1/fe)

    plt.subplot(2, 1, 2)
    plt.title('Réponse fréquentielle & Spectre du signal filtré')
    plt.plot(f, 20 * np.log10(np.abs(S_filtre)), label='Signal filtré')
    plt.plot(w_norm / (2 * np.pi), 20 * np.log10(np.abs(H)), label='Filtre ' + title)
    plt.xlim([fmin, fmax])
    plt.ylim([-Aa * 2, 0])
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Amplitude (dB)')
    for _fc in fc:
        plt.axvline(x=_fc-df/2, color='grey', linestyle='-.')
        plt.axvline(x=_fc+df/2, color='grey', linestyle='-.')
    plt.axhline(y=-Ap, color='b', linestyle='--')
    plt.axhline(y=-Aa, color='r', linestyle='--')
    plt.legend()
    plt.grid()

    # Ajoute de l'espace entre les deux sous-tracés
    plt.subplots_adjust(hspace=0.5)  # La valeur de hspace est comprise entre 0 et 1  

    plt.show()

    return


def plot_filter_frequential_responses(filters: tuple, fmin: float, fmax: float, Ap: float, Aa: float, fc: list, df: float) -> None:
    """
    Plot the filtered signal in time domain and the frequency response of the filter in amplitude in dB.
    :param filters: tuple of tuples of filters (w, H, title)
    :param fmin: minimum frequency
    :param fmax: maximum frequency
    :param Aa: stopband attenuation
    """
    # Affiche les réponses fréquentielles des filtres
    plt.figure('Réponses fréquentielles des filtres')

    # Plot the frequency response of the filter in amplitude in dB
    for w, H, title in filters:
        plt.plot(w/(2*np.pi), 20*np.log10(np.abs(H)), label=title)
    plt.xlim([fmin, fmax])
    plt.ylim([-Aa*2, 0])
    plt.xlabel('Frequence (Hz)')
    plt.ylabel('Amplitude (dB)')
    for _fc in fc:
        plt.axvline(x=_fc-df/2, color='grey', linestyle='-.')
        plt.axvline(x=_fc+df/2, color='grey', linestyle='-.')
    plt.axhline(y=-Ap, color='b', linestyle='--')
    plt.axhline(y=-Aa, color='r', linestyle='--')
    plt.legend()
    plt.grid()
    plt.show()

    return