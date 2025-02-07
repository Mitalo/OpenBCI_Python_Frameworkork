import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Especifica o caminho da pasta contendo os arquivos
# folder_path = "C:/Users/italo/OneDrive/Documentos/GitHub/OpenBCI_Python_Frameworkork/cues/coletas/treino novo/test"
folder_path = "C:/Users/italo/OneDrive/Documentos/TCC/Sinais filtrados para plotagem"

# Coleta os arquivos com "antonio" no nome dentro da pasta especificada
files_data = sorted(glob.glob(os.path.join(folder_path, "*_antonio_data.csv")))
files_events = sorted(glob.glob(os.path.join(folder_path, "*_antonio_events.csv")))

# Verifica se o número de arquivos é consistente
if len(files_data) != len(files_events):
    raise ValueError("O número de arquivos de dados e eventos não corresponde.")

# Itera sobre os arquivos para plotar os gráficos
for data_file, event_file in zip(files_data, files_events):
    # Lê os arquivos
    data = pd.read_csv(data_file)
    events = pd.read_csv(event_file)

    # Extrai as colunas relevantes
    signal_data_1 = data["Fp1"]  # Canal Fp1
    signal_data_2 = data["C3"]   # Canal C3
    marker_events = events["marker"]

    # Usa o índice das linhas como eixo x
    time_data = data.index
    time_events = events.index

    # Substitui "antonio" por "voluntario2" no nome do arquivo para o título
    title = os.path.basename(data_file).replace("antonio", "voluntario2")

    # Configuração da figura e subgráficos
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))  # Dois gráficos lado a lado

    # Considera apenas a metade central dos dados
    quarter_index = len(time_data) // 4
    three_quarter_index = 3 * len(time_data) // 4
    time_data_middle = time_data[quarter_index:three_quarter_index]
    signal_data_1_middle = signal_data_1[quarter_index:three_quarter_index]
    signal_data_2_middle = signal_data_2[quarter_index:three_quarter_index]
    time_events_middle = time_events[quarter_index:three_quarter_index]
    marker_events_middle = marker_events[quarter_index:three_quarter_index]

    # Plot do sinal Fp1 no primeiro gráfico
    ax1.plot(time_data_middle, signal_data_1_middle, label="Sinal (canal 1)", color="blue")
    ax1.set_ylabel("canal 1")
    ax1.set_title(f"{title} - canal 1")
    
    # Adiciona o eixo secundário para os eventos no gráfico do Fp1
    ax1_2 = ax1.twinx()
    ax1_2.plot(time_events_middle, marker_events_middle, label="Eventos", color="orange", linestyle="--")
    ax1_2.set_ylabel("Eventos")
    ax1.legend(loc="upper left")
    ax1_2.legend(loc="upper right")

    # Plot do sinal C3 no segundo gráfico
    ax2.plot(time_data_middle, signal_data_2_middle, label="Sinal (canal 2)", color="green")
    ax2.set_ylabel("canal 2")
    ax2.set_title(f"{title} - canal 2")
    
    # Adiciona o eixo secundário para os eventos no gráfico do C3
    ax2_2 = ax2.twinx()
    ax2_2.plot(time_events_middle, marker_events_middle, label="Eventos", color="orange", linestyle="--")
    ax2_2.set_ylabel("Eventos")
    ax2.legend(loc="upper left")
    ax2_2.legend(loc="upper right")

    # Configura o eixo X para ambos os gráficos
    ax1.set_xlabel("Tempo (time_board)")
    ax2.set_xlabel("Tempo (time_board)")

    # Exibe o gráfico
    plt.tight_layout()
    plt.show()