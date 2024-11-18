import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Função para carregar os dados do CSV
def load_data(file_path):
    return pd.read_csv(file_path)

# Função para aplicar um filtro passa-banda
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs  # Frequência de Nyquist (metade da frequência de amostragem)
    low = lowcut / nyquist  # Normaliza a frequência de corte inferior
    high = highcut / nyquist  # Normaliza a frequência de corte superior
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Função para filtrar o sinal
def bandpass_filter(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)

# Função para retificar o sinal
def rectify_signal(data):
    return np.abs(data)

# Função para aplicar média móvel no sinal
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# Função para normalizar o sinal
def normalize_signal(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Função para visualizar os sinais de EMG
def plot_signals(time, signal_fp1, signal_c3, title="Sinais EMG Processados"):
    plt.figure(figsize=(10, 6))
    
    # Fp1
    plt.subplot(2, 1, 1)
    plt.plot(time, signal_fp1, label="Fp1")
    plt.title(f'{title} - Fp1')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Amplitude Normalizada')
    plt.grid(True)

    # C3
    plt.subplot(2, 1, 2)
    plt.plot(time, signal_c3, label="C3", color='orange')
    plt.title(f'{title} - C3')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Amplitude Normalizada')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Carregar os dados
file_path = 'cues\coletas\A_antonio.csv'  # Substitua pelo caminho do seu arquivo
data = load_data(file_path)

# Definindo parâmetros
fs = 250  # Frequência de amostragem
lowcut = 20  # Frequência mínima do filtro (Hz)
highcut = 120  # Frequência máxima do filtro (Hz)
window_size = 50  # Tamanho da janela para média móvel

# Filtrando os sinais
signal_fp1 = bandpass_filter(data['Fp1'], lowcut, highcut, fs)
signal_c3 = bandpass_filter(data['C3'], lowcut, highcut, fs)

# Retificando os sinais
signal_fp1_rectified = rectify_signal(signal_fp1)
signal_c3_rectified = rectify_signal(signal_c3)

# Aplicando a média móvel
signal_fp1_smooth = moving_average(signal_fp1_rectified, window_size)
signal_c3_smooth = moving_average(signal_c3_rectified, window_size)

# Normalizando os sinais
signal_fp1_normalized = normalize_signal(signal_fp1_smooth)
signal_c3_normalized = normalize_signal(signal_c3_smooth)

# Definindo o tempo baseado no time_board
time = data['time_board'] / fs  # Assumindo que time_board está em amostras

# Visualizando os sinais processados
plot_signals(time, signal_fp1_normalized, signal_c3_normalized)
