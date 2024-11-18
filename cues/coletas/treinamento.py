import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Função para carregar os dados
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Função para aplicar o filtro Butterworth (banda passante)
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Função para filtrar o sinal
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

# Função de retificação do sinal (valor absoluto)
def rectify_signal(signal):
    return np.abs(signal)

# Função para média móvel
def moving_average(signal, window_size):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

# Função de normalização
def normalize_signal(signal):
    return (signal - np.mean(signal)) / np.std(signal)

# Funções para extração das características
def compute_mav(signal):
    return np.mean(np.abs(signal))

def compute_zc(signal):
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return len(zero_crossings)

def compute_ssc(signal):
    ssc_count = 0
    for i in range(1, len(signal)-1):
        if np.sign(signal[i] - signal[i-1]) != np.sign(signal[i+1] - signal[i]):
            ssc_count += 1
    return ssc_count

def compute_wl(signal):
    return np.sum(np.abs(np.diff(signal)))

def extract_features(signal, fs):
    mav = compute_mav(signal)
    zc = compute_zc(signal)
    ssc = compute_ssc(signal)
    wl = compute_wl(signal)
    energy = np.sum(signal**2) / len(signal)
    return [mav, zc, ssc, wl, energy]

# Função para processar e extrair características dos dados
def process_and_extract_features(data, fs, lowcut, highcut, window_size):
    signal_fp1 = bandpass_filter(data['Fp1'], lowcut, highcut, fs)
    signal_c3 = bandpass_filter(data['C3'], lowcut, highcut, fs)

    # Retificação
    signal_fp1_rectified = rectify_signal(signal_fp1)
    signal_c3_rectified = rectify_signal(signal_c3)

    # Média Móvel
    signal_fp1_smooth = moving_average(signal_fp1_rectified, window_size)
    signal_c3_smooth = moving_average(signal_c3_rectified, window_size)

    # Normalização
    signal_fp1_normalized = normalize_signal(signal_fp1_smooth)
    signal_c3_normalized = normalize_signal(signal_c3_smooth)

    features = []
    labels = []

    # Extrair características para cada janela de tempo
    for i in range(0, len(data), fs):
        window_fp1 = signal_fp1_normalized[i:i+fs]
        window_c3 = signal_c3_normalized[i:i+fs]

        if len(window_fp1) == fs and len(window_c3) == fs:
            combined_features = extract_features(np.concatenate([window_fp1, window_c3]), fs)
            features.append(combined_features)
            label = data['marker'].iloc[i]
            labels.append(label)

    return np.array(features), np.array(labels)

# Função para carregar e processar múltiplos arquivos
def process_multiple_files(file_paths, fs, lowcut, highcut, window_size):
    features_list = []
    labels_list = []
    
    for file_path in file_paths:
        data = load_data(file_path)
        features, labels = process_and_extract_features(data, fs, lowcut, highcut, window_size)
        features_list.append(features)
        labels_list.append(labels)
    
    # Concatenando todas as características e labels
    features_combined = np.vstack(features_list)
    labels_combined = np.concatenate(labels_list)
    
    return features_combined, labels_combined

# Função para treinar o modelo
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Função para avaliar o modelo
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # F1 Score ponderado
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'F1 Score: {f1:.2f}')

# Lista de arquivos de treino
train_files = [
    'cues/coletas/A_angelo.csv',
    'cues/coletas/E_angelo.csv',
    'cues/coletas/I_angelo.csv',
    'cues/coletas/O_angelo.csv', 
    'cues/coletas/U_angelo.csv', 
    'cues/coletas/A_gabriel.csv',
    'cues/coletas/E_gabriel.csv', 
    'cues/coletas/I_gabriel.csv', 
    'cues/coletas/O_gabriel.csv',
    'cues/coletas/U_gabriel.csv', 
    'cues/coletas/A_gustavo.csv', 
    'cues/coletas/E_gustavo.csv',
    'cues/coletas/I_gustavo.csv', 
    'cues/coletas/O_gustavo.csv', 
    'cues/coletas/U_gustavo.csv',
    'cues/coletas/A_mateus.csv', 
    'cues/coletas/E_mateus.csv', 
    'cues/coletas/I_mateus.csv',
    'cues/coletas/O_mateus.csv', 
    'cues/coletas/U_mateus.csv', 
    'cues/coletas/A_murilo.csv',
    'cues/coletas/E_murilo.csv', 
    'cues/coletas/I_murilo.csv', 
    'cues/coletas/O_murilo.csv',
    'cues/coletas/U_murilo.csv', 
    'cues/coletas/A_stefanye.csv', 
    'cues/coletas/E_stefanye.csv',
    'cues/coletas/I_stefanye.csv', 
    'cues/coletas/O_stefanye.csv', 
    'cues/coletas/U_stefanye.csv'
]

# Lista de arquivos de teste
test_files = [
    'cues/coletas/A_antonio.csv', 
    'cues/coletas/E_antonio.csv', 
    'cues/coletas/I_antonio.csv',
    'cues/coletas/O_antonio.csv', 
    'cues/coletas/U_antonio.csv'
]

# Definir parâmetros
fs = 250  # Frequência de amostragem
lowcut = 20.0  # Frequência de corte inferior
highcut = 120.0  # Frequência de corte superior (ajustado para 120 Hz)
window_size = 50  # Tamanho da janela para média móvel

# Processar os dados de treino
features_train, labels_train = process_multiple_files(train_files, fs, lowcut, highcut, window_size)

# Treinar o modelo com todos os dados de treino
model = train_model(features_train, labels_train)

# Processar os dados de teste
features_test, labels_test = process_multiple_files(test_files, fs, lowcut, highcut, window_size)

# Avaliar o modelo nos dados de teste
print("Avaliação no conjunto de teste:")
evaluate_model(model, features_test, labels_test)
