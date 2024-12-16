import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from collections import Counter
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier


# Função para carregar os dados
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Função para aplicar o filtro Butterworth (banda passante)
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
    
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def apply_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Função para filtrar o sinal
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

def apply_notch_filter(data, fs, f0=60.0, Q=30.0):
    b, a = iirnotch(f0, Q, fs)
    y = filtfilt(b, a, data)
    return y

# Função de retificação do sinal (valor absoluto)
def rectify_signal(signal):
    return np.abs(signal)

# Função para média móvel
def moving_average(signal, window_size):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

# Função de normalização
def normalize_signal(signal):
    return (signal - np.mean(signal)) / np.std(signal)

def min_max(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

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
    
def compute_dasdv(signal):
    diff = np.diff(signal)
    abs_diff = np.abs(diff)
    mean_squares = np.mean(np.square(abs_diff))
    dasdv_value = np.sqrt(mean_squares)
    return dasdv_value

def compute_wamp(signal):
    wamp = 0
    for i in range(1, len(signal)):
        if abs(signal[i] - signal[i-1]) > 0:
            wamp += 1
    return wamp
        
def extract_features(signal, fs):
    mav = compute_mav(signal)
    zc = compute_zc(signal)
    ssc = compute_ssc(signal)
    wl = compute_wl(signal)
    wamp = compute_wamp(signal)
    dasdv = compute_dasdv(signal)
    return [mav, zc, ssc, wl]

# Função para processar e extrair características dos dados
def process_and_extract_features(signal_data, event_data, fs, lowcut, highcut, window_size):
    # signal_fp1 = bandpass_filter(signal_data['Fp1'], lowcut, highcut, fs)
    # signal_c3 = bandpass_filter(signal_data['C3'], lowcut, highcut, fs)

    signal_fp1 = apply_highpass_filter(signal_data['Fp1'], 30, fs)
    signal_c3 = apply_highpass_filter(signal_data['C3'], 30, fs)

    # signal_fp1 = apply_lowpass_filter(signal_fp1, 100, fs)
    # signal_c3 = apply_lowpass_filter(signal_c3, 100, fs)

    signal_fp1 = apply_notch_filter(signal_fp1, fs)
    signal_c3 = apply_notch_filter(signal_c3, fs)

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
    for i in range(0, len(signal_data), 5*fs):
        window_fp1 = signal_fp1_normalized[i:i+5*fs]
        window_c3 = signal_c3_normalized[i:i+5*fs]

        # if len(window_fp1) == 5*fs and len(window_c3) == 5*fs:
        combined_features = extract_features(np.concatenate([window_fp1, window_c3]), fs)
        features.append(combined_features)

    for i in range(0, len(event_data), 5*fs):
        window_label = event_data[i:i+5*fs]['marker']
        
        counter = Counter(window_label)
        label,_ = counter.most_common(1)[0]#round(np.average(window_label))
        labels.append(label)

        # if len(window_label) == 5*fs:
        #     label = window_label['marker'].iloc[i]
        #     labels.append(label)

    return np.array(features), np.array(labels)

# Função para carregar e processar múltiplos arquivos
def process_multiple_files(signal_files, event_files, fs, lowcut, highcut, window_size):
    features_list = []
    labels_list = []
    
    for signal_file, event_file in zip(signal_files, event_files):
        signal_data = load_data(signal_file)
        event_data = load_data(event_file)
        features, labels = process_and_extract_features(signal_data, event_data, fs, lowcut, highcut, window_size)
        features_list.append(features)
        labels_list.append(labels)

    # Concatenando todas as características e labels
    features_combined = min_max(np.vstack(features_list))
    labels_combined = np.concatenate(labels_list)
    
    return features_combined, labels_combined

# Função para treinar o modelo
def train_model(X_train, y_train):
    model = LinearDiscriminantAnalysis()#SVC(kernel='linear', C=1.0, random_state=42)
    model.fit(X_train, y_train)
    return model
    # classifiers = {
        # 'SVC':{
        #     'model': SVC(),
        #     'params': {
        #         'C': [0.1, 1, 10, 100, 1000],
        #         'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        #         'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        #         'degree': [2, 3, 4, 5]
        #     }
        # },
        # 'RandomForest':{
        #     'model': RandomForestClassifier(),
        #     'params': {
        #         'n_estimators': [10, 50, 100, 200, 500],
        #         'max_features': ['auto', 'sqrt', 'log2'],
        #         'max_depth' : [4,5,6,7,8],
        #         'criterion' :['gini', 'entropy']
        #     }
        # },
        # 'logistic_regression':{
        #     'model': LogisticRegression(),
        #     'params': {
        #         'C': [0.1, 1, 10, 100, 1000],
        #         'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        #     }
        # },
        # 'lda':{
        #     'model': LinearDiscriminantAnalysis(),
        #     'params': {
        #         'solver': ['svd', 'lsqr', 'eigen']
        #     }
        # }
        # 'qda':{
        #     'model': QuadraticDiscriminantAnalysis(),
        #     'params': {
        #         'reg_param': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        #     }
        # },
        # 'knn':{
        #     'model': KNeighborsClassifier(),
        #     'params': {
        #         'n_neighbors': [3, 5, 7, 9, 11],
        #         'weights': ['uniform', 'distance'],
        #         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        #     }
        # },
        # 'ada':{
        #     'model': AdaBoostClassifier(),
        #     'params': {
        #         'n_estimators': [50, 100, 200, 500],
        #         'learning_rate': [0.001, 0.01, 0.1, 1, 10]
        #     }
        # },
        # 'bagging':{
        #     'model': BaggingClassifier(),
        #     'params': {
        #         'n_estimators': [10, 50, 100, 200, 500],
        #         'max_samples': [1, 2, 3, 4, 5],
        #         'max_features': [1, 2, 3, 4, 5]
        #     }
        # },
        # 'extra_trees':{
        #     'model': ExtraTreesClassifier(),
        #     'params': {
        #         'n_estimators': [10, 50, 100, 200, 500],
        #         'max_features': ['auto', 'sqrt', 'log2'],
        #         'max_depth' : [4,5,6,7,8],
        #         'criterion' :['gini', 'entropy']
        #     }
        # },
        # 'gradient_boosting':{
        #     'model': GradientBoostingClassifier(),
        #     'params': {
        #         'n_estimators': [10, 50, 100, 200, 500],
        #         'learning_rate': [0.001, 0.01, 0.1, 1, 10],
        #         'subsample': [0.1, 0.5, 1.0],
        #         'max_depth' : [4,5,6,7,8]
        #     }
        # },
        # 'voting':{
        #     'model': VotingClassifier(estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier()), ('svc', SVC())]),
        #     'params': {
        #         'voting': ['hard', 'soft']
        #     }
        # },
        # 'stacking':{
        #     'model': StackingClassifier(estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier()), ('svc', SVC())]),
        #     'params': {
        #         'final_estimator': [LogisticRegression(), RandomForestClassifier(), SVC()]
        #     }
        # },
        # 'nearest_centroid':{
        #     'model': NearestCentroid(),
        #     'params': {
        #         'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
        #         'shrink_threshold': [None, 0.1, 0.2, 0.3, 0.4, 0.5]
        #     }
        # },
        # 'radius_neighbors':{
        #     'model': RadiusNeighborsClassifier(),
        #     'params': {
        #         'radius': [0.5, 1.0, 1.5, 2.0, 2.5],
        #         'weights': ['uniform', 'distance'],
        #         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        #     }
        # }
    # }
    
    # best_models = {}
    # highest_accuracy = 0
    # best_model_name = ""

    # for model_name, mp in classifiers.items():
    #     clf = GridSearchCV(estimator=mp['model'], param_grid=mp['params'], cv=5, verbose=2, n_jobs=-1, return_train_score=False)
    #     clf.fit(X_train, y_train)
    #     best_models[model_name] = clf
    #     if clf.best_score_ > highest_accuracy:
    #         highest_accuracy = clf.best_score_
    #         best_model_name = model_name

    # print(f"Melhor modelo: {best_model_name}")
    # print(f"Melhores parâmetros: {best_models[best_model_name].best_params_}")
    # print(f"Melhor acurácia: {highest_accuracy}")

    # return best_models[best_model_name]

# Função para avaliar o modelo
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # F1 Score ponderado
    print(f'Accuracy: {accuracy * 100:.5f}%')
    print(f'F1 Score: {f1:.2f}')

# Lista de arquivos de treino (sinais e eventos)
train_signal_files = [
    'cues/coletas/treino novo/train/A_angelo_data.csv',
    'cues/coletas/treino novo/train/E_angelo_data.csv',
    'cues/coletas/treino novo/train/I_angelo_data.csv',
    'cues/coletas/treino novo/train/O_angelo_data.csv', 
    'cues/coletas/treino novo/train/U_angelo_data.csv', 
    'cues/coletas/treino novo/train/A_gabriel_data.csv',
    'cues/coletas/treino novo/train/E_gabriel_data.csv', 
    'cues/coletas/treino novo/train/I_gabriel_data.csv', 
    'cues/coletas/treino novo/train/O_gabriel_data.csv',
    'cues/coletas/treino novo/train/U_gabriel_data.csv', 
    'cues/coletas/treino novo/train/A_gustavo_data.csv', 
    'cues/coletas/treino novo/train/E_gustavo_data.csv',
    'cues/coletas/treino novo/train/I_gustavo_data.csv', 
    'cues/coletas/treino novo/train/O_gustavo_data.csv', 
    'cues/coletas/treino novo/train/U_gustavo_data.csv',
    'cues/coletas/treino novo/train/A_mateus_data.csv', 
    'cues/coletas/treino novo/train/E_mateus_data.csv', 
    'cues/coletas/treino novo/train/I_mateus_data.csv',
    'cues/coletas/treino novo/train/O_mateus_data.csv', 
    'cues/coletas/treino novo/train/U_mateus_data.csv', 
    'cues/coletas/treino novo/train/A_messias_data.csv', 
    'cues/coletas/treino novo/train/E_messias_data.csv', 
    'cues/coletas/treino novo/train/I_messias_data.csv',
    'cues/coletas/treino novo/train/O_messias_data.csv', 
    'cues/coletas/treino novo/train/U_messias_data.csv', 
    'cues/coletas/treino novo/train/A_murilo_data.csv',
    'cues/coletas/treino novo/train/E_murilo_data.csv', 
    'cues/coletas/treino novo/train/I_murilo_data.csv', 
    'cues/coletas/treino novo/train/O_murilo_data.csv',
    'cues/coletas/treino novo/train/U_murilo_data.csv', 
    'cues/coletas/treino novo/train/A_stefanye_data.csv', 
    'cues/coletas/treino novo/train/E_stefanye_data.csv',
    'cues/coletas/treino novo/train/I_stefanye_data.csv', 
    'cues/coletas/treino novo/train/O_stefanye_data.csv', 
    'cues/coletas/treino novo/train/U_stefanye_data.csv'
]

train_event_files = [
    'cues/coletas/treino novo/train/A_angelo_events.csv',
    'cues/coletas/treino novo/train/E_angelo_events.csv',
    'cues/coletas/treino novo/train/I_angelo_events.csv',
    'cues/coletas/treino novo/train/O_angelo_events.csv', 
    'cues/coletas/treino novo/train/U_angelo_events.csv', 
    'cues/coletas/treino novo/train/A_gabriel_events.csv',
    'cues/coletas/treino novo/train/E_gabriel_events.csv', 
    'cues/coletas/treino novo/train/I_gabriel_events.csv', 
    'cues/coletas/treino novo/train/O_gabriel_events.csv',
    'cues/coletas/treino novo/train/U_gabriel_events.csv',
    'cues/coletas/treino novo/train/A_gustavo_events.csv', 
    'cues/coletas/treino novo/train/E_gustavo_events.csv',
    'cues/coletas/treino novo/train/I_gustavo_events.csv', 
    'cues/coletas/treino novo/train/O_gustavo_events.csv', 
    'cues/coletas/treino novo/train/U_gustavo_events.csv',
    'cues/coletas/treino novo/train/A_mateus_events.csv', 
    'cues/coletas/treino novo/train/E_mateus_events.csv', 
    'cues/coletas/treino novo/train/I_mateus_events.csv',
    'cues/coletas/treino novo/train/O_mateus_events.csv', 
    'cues/coletas/treino novo/train/U_mateus_events.csv', 
    'cues/coletas/treino novo/train/A_messias_events.csv', 
    'cues/coletas/treino novo/train/E_messias_events.csv', 
    'cues/coletas/treino novo/train/I_messias_events.csv',
    'cues/coletas/treino novo/train/O_messias_events.csv', 
    'cues/coletas/treino novo/train/U_messias_events.csv', 
    'cues/coletas/treino novo/train/A_murilo_events.csv',
    'cues/coletas/treino novo/train/E_murilo_events.csv', 
    'cues/coletas/treino novo/train/I_murilo_events.csv', 
    'cues/coletas/treino novo/train/O_murilo_events.csv',
    'cues/coletas/treino novo/train/U_murilo_events.csv', 
    'cues/coletas/treino novo/train/A_stefanye_events.csv', 
    'cues/coletas/treino novo/train/E_stefanye_events.csv',
    'cues/coletas/treino novo/train/I_stefanye_events.csv', 
    'cues/coletas/treino novo/train/O_stefanye_events.csv', 
    'cues/coletas/treino novo/train/U_stefanye_events.csv'
]

# Lista de arquivos de teste (sinais e eventos)
test_signal_files = [
    'cues/coletas/treino novo/test/A_antonio_data.csv', 
    'cues/coletas/treino novo/test/E_antonio_data.csv', 
    'cues/coletas/treino novo/test/I_antonio_data.csv',
    'cues/coletas/treino novo/test/O_antonio_data.csv', 
    'cues/coletas/treino novo/test/U_antonio_data.csv'
]

test_event_files = [
    'cues/coletas/treino novo/test/A_antonio_events.csv', 
    'cues/coletas/treino novo/test/E_antonio_events.csv', 
    'cues/coletas/treino novo/test/I_antonio_events.csv',
    'cues/coletas/treino novo/test/O_antonio_events.csv', 
    'cues/coletas/treino novo/test/U_antonio_events.csv'
]

# Definir parâmetros
fs = 250  # Frequência de amostragem
lowcut = 15.0  # Frequência de corte inferior
highcut = 120.0  # Frequência de corte superior (ajustado para 120 Hz)
window_size = 20  # Tamanho da janela para média móvel

# Processar os dados de treino
features_train, labels_train = process_multiple_files(train_signal_files, train_event_files, fs, lowcut, highcut, window_size)

# Treinar o modelo com todos os dados de treino
model = train_model(features_train, labels_train)

# Processar os dados de teste
features_test, labels_test = process_multiple_files(test_signal_files, test_event_files, fs, lowcut, highcut, window_size)

# Avaliar o modelo nos dados de teste
print("Avaliação no conjunto de teste:")
evaluate_model(model, features_test, labels_test)

# predictions = model.predict(features_test)

# output_df = pd.DataFrame({
#     'True Label': labels_test,
#     'Predicted Label': predictions
# })
# output_df.to_csv('G:/Meu Drive/PDI/UTFPR/TCC/Outputs/predictions.csv', index=False)
