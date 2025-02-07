import pandas as pd
import os

# Função para dividir os arquivos CSV

def dividir_csv_em_massa(input_dir, coluna_marcador, movimentos_por_arquivo, output_dir):
    # Listar todos os arquivos na pasta de entrada
    arquivos = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    if not arquivos:
        print("Nenhum arquivo CSV encontrado na pasta especificada.")
        return

    # Criar o diretório de saída, se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Processar cada arquivo
    for arquivo in arquivos:
        arquivo_entrada = os.path.join(input_dir, arquivo)

        # Ler o arquivo CSV
        dados = pd.read_csv(arquivo_entrada)

        # Garantir que a coluna marcador está presente
        if coluna_marcador not in dados.columns:
            print(f"A coluna '{coluna_marcador}' não foi encontrada no arquivo {arquivo}. Pulando este arquivo.")
            continue

        # Modificar os valores da coluna marcador
        dados[coluna_marcador] = dados[coluna_marcador] / 1000 - 1

        # Identificar os pontos de troca do marcador
        trocas = dados[coluna_marcador].ne(dados[coluna_marcador].shift()).cumsum()

        # Adicionar uma coluna de grupo baseada nas transições do marcador
        dados['grupo'] = (trocas - 1) // (2 * movimentos_por_arquivo)  # Cada movimento alterna marcador 2 vezes

        # Obter o nome base do arquivo de entrada (sem extensão)
        base_nome_arquivo = os.path.splitext(arquivo)[0]

        # Dividir os dados em arquivos
        grupos = dados['grupo'].unique()
        for grupo in grupos:
            subset = dados[dados['grupo'] == grupo].drop(columns=['grupo'])  # Remover coluna auxiliar

            # Criar dois arquivos separados
            data_subset = subset[["Fp1", "C3", "time_board"]]
            events_subset = subset[["time_board", coluna_marcador]]

            # Nomes dos arquivos gerados
            data_nome_arquivo = os.path.join(output_dir, f"{base_nome_arquivo}_{grupo + 1}_data.csv")
            events_nome_arquivo = os.path.join(output_dir, f"{base_nome_arquivo}_{grupo + 1}_events.csv")

            # Salvar os arquivos
            data_subset.to_csv(data_nome_arquivo, index=False)
            events_subset.to_csv(events_nome_arquivo, index=False)

            print(f"Arquivos criados: {data_nome_arquivo}, {events_nome_arquivo}")

# Exemplo de uso
dividir_csv_em_massa(
    input_dir="C:/Users/italo/OneDrive/Documentos/TCC/coletas",           # Pasta onde estão os arquivos CSV
    coluna_marcador="marker",         # Coluna que contém o marcador
    movimentos_por_arquivo=5,          # Número de movimentos por arquivo
    output_dir="C:/Users/italo/OneDrive/Documentos/TCC/novos dados" # Diretório onde os arquivos serão salvos
)


#### CONFIGURATION:

# {
#     "show_diagram": false,
#     "nodes": {
#       "root": {
#         "data": {
#           "module": "models.node.generator.file",
#           "type": "CSVFileArray",
#           "enable_log": true,
#           "file_path":
#           [
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\A_angelo_data.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\E_angelo_data.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\I_angelo_data.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\O_angelo_data.csv", 
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\U_angelo_data.csv", 
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\A_gabriel_data.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\E_gabriel_data.csv", 
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\I_gabriel_data.csv", 
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\O_gabriel_data.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\U_gabriel_data.csv", 
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\A_gustavo_data.csv", 
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\E_gustavo_data.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\I_gustavo_data.csv", 
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\O_gustavo_data.csv", 
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\U_gustavo_data.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\A_mateus_data.csv", 
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\E_mateus_data.csv", 
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\I_mateus_data.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\O_mateus_data.csv", 
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\U_mateus_data.csv", 
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\A_messias_data.csv", 
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\E_messias_data.csv", 
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\I_messias_data.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\O_messias_data.csv", 
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\U_messias_data.csv", 
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\A_murilo_data.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\E_murilo_data.csv", 
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\I_murilo_data.csv", 
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\O_murilo_data.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\U_murilo_data.csv", 
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\A_stefanye_data.csv", 
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\E_stefanye_data.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\I_stefanye_data.csv", 
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\O_stefanye_data.csv", 
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\U_stefanye_data.csv"
#           ],
#           "data_type": "float",
#           "sampling_frequency": 250,
#           "channel_column_names": [
#             "Fp1",
#             "C3"
#             ],
#           "buffer_options":{
#             "clear_output_buffer_on_generate": true,
#             "print_buffer_size": false
#           },
#           "outputs": {
#             "main": [
#               {
#                 "node": "lowpass",
#                 "input": "main"
#               }
#             ],
#             "timestamp": []
#           }
#         },
#         "events": {
#           "module": "models.node.generator.file",
#           "type": "CSVFileArray",
#           "enable_log": true,
#           "file_path": [
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\A_angelo_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\E_angelo_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\I_angelo_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\O_angelo_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\U_angelo_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\A_gabriel_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\E_gabriel_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\I_gabriel_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\O_gabriel_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\U_gabriel_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\A_gustavo_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\E_gustavo_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\I_gustavo_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\O_gustavo_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\U_gustavo_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\A_mateus_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\E_mateus_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\I_mateus_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\O_mateus_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\U_mateus_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\A_murilo_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\E_murilo_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\I_murilo_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\O_murilo_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\U_murilo_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\A_messias_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\E_messias_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\I_messias_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\O_messias_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\U_messias_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\A_stefanye_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\E_stefanye_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\I_stefanye_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\O_stefanye_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\train\\U_stefanye_events.csv"
#           ],
#           "sampling_frequency": 250,
#           "channel_column_names": [
#             "marker"
#           ],
#           "buffer_options": {
#             "clear_output_buffer_on_generate": true,
#             "print_buffer_size": false
#           },
#           "outputs": {
#             "main": [
#               {
#                 "node": "feature_extraction",
#                 "input": "events"
#               }
#             ],
#             "timestamp": []
#           }
#         }
#       },
#       "common": {
#         "lowpass": {
#           "module": "models.node.processing.filter",
#           "type": "LowPass",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "high_cut_frequency_hz": 120.0,
#           "order": 4,
#           "outputs": {
#             "main": [
#               {
#                 "node": "highpass",
#                 "input": "main"
#               }
#             ]
#           }
#         },
#         "highpass": {
#           "module": "models.node.processing.filter",
#           "type": "HighPass",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "low_cut_frequency_hz": 30.0,
#           "order": 2,
#           "outputs": {
#             "main": [
#               {
#                 "node": "bandpass",
#                 "input": "main"
#               }
#             ]
#           }
#         },
#         "bandpass": {
#           "module": "models.node.processing.filter",
#           "type": "BandPass",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "low_cut_frequency_hz": 30.0,
#           "high_cut_frequency_hz": 120.0,
#           "order": 3,
#           "outputs": {
#             "main": [
#               {
#                 "node": "notch",
#                 "input": "main"
#               }
#             ]
#           }
#         },
#         "notch": {
#           "module": "models.node.processing.filter",
#           "type": "Notch",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "cut_frequency_hz": 60.0,
#           "Q": 60,
#           "outputs": {
#             "main": [
#               {
#                 "node": "feature_extraction",
#                 "input": "features"
#               }
#             ]
#           }
#         },
#         "rectify": {
#           "module": "models.node.processing",
#           "type": "RectifySignal",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "outputs": {
#             "main": [
#               {
#                 "node": "movingAverage",
#                 "input": "main"
#               }
#             ]
#           }
#         },
#         "movingAverage": {
#           "module": "models.node.processing",
#           "type": "Smoothing",
#           "window_type": "exponential",
#           "window_size": 50,
#           "convolution_mode": "same",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "outputs": {
#             "main": [
#               {
#                 "node": "normalize",
#                 "input": "main"
#               }
#             ]
#           }
#         },
#         "normalize": {
#           "module": "models.node.processing",
#           "type": "NormalizeSignal",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "outputs": {
#             "main": [
#               {
#                 "node": "feature_extraction",
#                 "input": "features"
#               }
#             ]
#           }
#         },
#         "segmentation": {
#           "module": "models.node.processing.segmenter",
#           "type": "LabelBasedFixedWindowSegmenter",
#           "enable_log": true,
#           "label_value": 0,
#           "filling_value": "zero",
#           "samples_after_label": 0,
#           "samples_before_label": 1250,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": false,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "outputs": {
#             "data": [
#               {
#                 "node": "feature_extraction",
#                 "input": "features"
#               }
#             ],
#             "label": [
#               {
#                 "node": "feature_extraction",
#                 "input": "events"
#               }
#             ]
#           }
#         },
#         "feature_extraction": {
#           "module": "models.node.processing",
#           "type": "FeatureExtractionNode",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "sampling_frequency": 250,
#           "outputs": {
#             "features": [
#               {
#                 "node": "lda",
#                 "input": "data"
#               }
#             ],
#             "events": [
#               {
#                 "node": "lda",
#                 "input": "label"
#               }
#             ]
#           }
#         },
#         "lda": {
#           "module": "models.node.processing.trainable.classifier",
#           "type": "LDA",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "clear_input_buffer_after_training": true,
#             "process_input_buffer_after_training": false,
#             "print_buffer_size": false
#           },
#           "training_set_size": 2,
#           "save_after_training": true,
#           "load_trained": false,
#           "save_file_path": "C:\\Users\\italo\\OneDrive\\Documentos\\TCC\\novos dadosassificadores\\LDA_emg.sav",
#           "load_file_path": "C:\\Users\\italo\\OneDrive\\Documentos\\TCC\\novos dadosassificadores\\LDA_emg.sav",
#           "outputs": {
#             "main": [],
#             "probability": [],
#             "training_finished": []
#           }
#         },
#         "logistic": {
#           "module": "models.node.processing.trainable.classifier",
#           "type": "logitregression",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "clear_input_buffer_after_training": true,
#             "process_input_buffer_after_training": false,
#             "print_buffer_size": false
#           },
#           "training_set_size": 2,
#           "save_after_training": true,
#           "load_trained": false,
#           "save_file_path": "C:\\Users\\italo\\OneDrive\\Documentos\\TCC\\novos dadosassificadores\\LR_emg.sav",
#           "load_file_path": "C:\\Users\\italo\\OneDrive\\Documentos\\TCC\\novos dadosassificadores\\LR_emg.sav",
#           "outputs": {
#             "main": [],
#             "probability": [],
#             "training_finished": []
#           }
#         },
#         "svc": {
#           "module": "models.node.processing.trainable.classifier",
#           "type": "SVC",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "clear_input_buffer_after_training": true,
#             "process_input_buffer_after_training": false,
#             "print_buffer_size": false
#           },
#           "training_set_size": 2,
#           "save_after_training": true,
#           "load_trained": false,
#           "save_file_path": "C:\\Users\\italo\\OneDrive\\Documentos\\TCC\\novos dadosassificadores\\SVC_emg.sav",
#           "load_file_path": "C:\\Users\\italo\\OneDrive\\Documentos\\TCC\\novos dadosassificadores\\SVC_emg.sav",
#           "outputs": {
#             "main": [],
#             "probability": [],
#             "training_finished": []
#           }
#         }
#       }
#     }
#   }


#### TEST:

# {
#     "show_diagram": false,
#     "nodes": {
#       "root": 
#       {
#        "test_emg": {
#           "module": "models.node.generator.file",
#           "type": "CSVFileArray",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_generate": true,
#             "print_buffer_size": false
#           },
#           "file_path": [
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\test\\A_antonio_data.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\test\\E_antonio_data.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\test\\I_antonio_data.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\test\\O_antonio_data.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\test\\U_antonio_data.csv"
#           ],
#           "data_type": "float",
#           "sampling_frequency": 250,
#           "channel_column_names": [
#             "Fp1",
#             "C3"
#           ],
#           "outputs": {
#             "main": [
#               {
#                 "node": "lowpass",
#                 "input": "main"
#               }
#             ],
#             "timestamp": []
#           }
#         },
#         "test_labels": {
#           "module": "models.node.generator.file",
#           "type": "CSVFileArray",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_generate": true,
#             "print_buffer_size": false
#           },
#           "file_path": [
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\test\\A_antonio_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\test\\E_antonio_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\test\\I_antonio_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\test\\O_antonio_events.csv",
#             "C:\\Users\\italo\\OneDrive\\Documentos\\GitHub\\OpenBCI_Python_Frameworkork\\cues\\coletas\\treino novo\\test\\U_antonio_events.csv"
#           ],
#           "data_type": "float",
#           "sampling_frequency": 250,
#           "channel_column_names": [
#             "marker"
#           ],
#           "outputs": {
#             "main": [
#                 {
#                 "node": "feature_extraction",
#                 "input": "events"
#                 }
#             ],
#             "timestamp": []
#           }
#         }
#       },
#       "common": 
#       {
#         "lowpass": {
#           "module": "models.node.processing.filter",
#           "type": "LowPass",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "high_cut_frequency_hz": 120.0,
#           "order": 4,
#           "outputs": {
#             "main": [
#               {
#                 "node": "highpass",
#                 "input": "main"
#               }
#             ]
#           }
#         },
#         "highpass": {
#           "module": "models.node.processing.filter",
#           "type": "HighPass",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "low_cut_frequency_hz": 30.0,
#           "order": 2,
#           "outputs": {
#             "main": [
#               {
#                 "node": "bandpass",
#                 "input": "main"
#               }
#             ]
#           }
#         },
#         "bandpass": {
#           "module": "models.node.processing.filter",
#           "type": "BandPass",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "low_cut_frequency_hz": 30.0,
#           "high_cut_frequency_hz": 120.0,
#           "order": 3,
#           "outputs": {
#             "main": [
#               {
#                 "node": "notch",
#                 "input": "main"
#               }
#             ]
#           }
#         },
#         "notch": {
#           "module": "models.node.processing.filter",
#           "type": "Notch",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "cut_frequency_hz": 60.0,
#           "Q": 60,
#           "outputs": {
#             "main": [
#               {
#                 "node": "feature_extraction",
#                 "input": "features"
#               }
#             ]
#           }
#         },
#         "rectify": {
#           "module": "models.node.processing",
#           "type": "RectifySignal",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "outputs": {
#             "main": [
#               {
#                 "node": "movingAverage",
#                 "input": "main"
#               }
#             ]
#           }
#         },
#         "movingAverage": {
#           "module": "models.node.processing",
#           "type": "Smoothing",
#           "window_type": "exponential",
#           "window_size": 50,
#           "convolution_mode": "same",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "outputs": {
#             "main": [
#               {
#                 "node": "normalize",
#                 "input": "main"
#               }
#             ]
#           }
#         },
#         "normalize": {
#           "module": "models.node.processing",
#           "type": "NormalizeSignal",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "outputs": {
#             "main": [
#               {
#                 "node": "feature_extraction",
#                 "input": "features"
#               }
#             ]
#           }
#         },
#         "feature_extraction": {
#           "module": "models.node.processing",
#           "type": "FeatureExtractionNode",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "sampling_frequency": 250,
#           "outputs": {
#             "features": [
#               {
#                 "node": "lda",
#                 "input": "data"
#               }
#             ],
#             "events": [
#               {
#                 "node": "accuracy",
#                 "input": "actual"
#               }
#             ]
#           }
#         },
#         "lda": {
#           "module": "models.node.processing.trainable.classifier",
#           "type": "LDA",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "clear_input_buffer_after_training": false,
#             "process_input_buffer_after_training": true,
#             "print_buffer_size": false
#           },
#           "training_set_size": 2,
#           "save_after_training": true,
#           "load_trained": true,
#           "save_file_path": "C:\\Users\\italo\\OneDrive\\Documentos\\TCC\\Classificadores\\LDA_emg.sav",
#           "load_file_path": "C:\\Users\\italo\\OneDrive\\Documentos\\TCC\\Classificadores\\LDA_emg.sav",
#           "outputs": {
#             "main": [],
#             "probability": [
#                 {
#                     "node": "decoder",
#                     "input": "main"
#                 }
#             ],
#             "training_finished": []
#           }
#         },
#         "logistic": {
#           "module": "models.node.processing.trainable.classifier",
#           "type": "logitregression",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "clear_input_buffer_after_training": false,
#             "process_input_buffer_after_training": true,
#             "print_buffer_size": false
#           },
#           "training_set_size": 2,
#           "save_after_training": true,
#           "load_trained": true,
#           "save_file_path": "C:\\Users\\italo\\OneDrive\\Documentos\\TCC\\Classificadores\\LR_emg.sav",
#           "load_file_path": "C:\\Users\\italo\\OneDrive\\Documentos\\TCC\\Classificadores\\LR_emg.sav",
#           "outputs": {
#             "main": [],
#             "probability": [
#                 {
#                     "node": "decoder",
#                     "input": "main"
#                 }
#             ],
#             "training_finished": []
#           }
#         },
#         "svc": {
#           "module": "models.node.processing.trainable.classifier",
#           "type": "SVC",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "clear_input_buffer_after_training": false,
#             "process_input_buffer_after_training": true,
#             "print_buffer_size": false
#           },
#           "training_set_size": 2,
#           "save_after_training": true,
#           "load_trained": true,
#           "save_file_path": "C:\\Users\\italo\\OneDrive\\Documentos\\TCC\\Classificadores\\SVC_emg.sav",
#           "load_file_path": "C:\\Users\\italo\\OneDrive\\Documentos\\TCC\\Classificadores\\SVC_emg.sav",
#           "outputs": {
#             "main": [],
#             "probability": [
#                 {
#                     "node": "decoder",
#                     "input": "main"
#                 }
#             ],
#             "training_finished": []
#           }
#         },
#         "accuracy": {
#           "module": "models.node.processing.metric",
#           "type": "accuracy",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "outputs": {
#             "main": [
#               {
#                 "node": "console_out",
#                 "input": "main"
#               }
#             ]
#           }
#         },
#         "console_out": {
#           "module": "models.node.output.display",
#           "type": "Console",
#           "prefix": "Accuracy Score = ",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "outputs": {}
#         },
#         "label_encoder": {
#           "module": "models.node.processing.encoder",
#           "type": "SingleToOneHot",
#           "enable_log": true,
#           "labels": [
#             "DESCANSO",
#             "A",
#             "E",
#             "I",
#             "O",
#             "U"
#           ],
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": true,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "outputs": {
#             "main": [
#               {
#                 "node": "accuracy",
#                 "input": "actual"
#               }
#             ]
#           }
#         },
#         "decoder": {
#           "module": "models.node.processing.encoder",
#           "type": "OneHotToSingle",
#           "enable_log": true,
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": false,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "outputs": {
#             "main": [
#               {
#                 "node": "accuracy",
#                 "input": "predicted"
#               }
#             ]
#           }
#         },
#         "csv_out": {
#           "module": "models.node.output.file",
#           "type": "CSVFile",
#           "enable_log": true,
#           "file_path": "C:\\Users\\italo\\OneDrive\\Documentos\\TCC\\Classificadores\\saida_LDA.csv",
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": false,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "outputs": {}
#         },
#         "csv_out2": {
#           "module": "models.node.output.file",
#           "type": "CSVFile",
#           "enable_log": true,
#           "file_path": "C:\\Users\\italo\\OneDrive\\Documentos\\TCC\\Classificadores\\saida_one_to_sing.csv",
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": false,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "outputs": {}
#         },
#         "csv_out3": {
#           "module": "models.node.output.file",
#           "type": "CSVFile",
#           "enable_log": true,
#           "file_path": "C:\\Users\\italo\\OneDrive\\Documentos\\TCC\\Classificadores\\saida_feat_extract.csv",
#           "buffer_options": {
#             "clear_output_buffer_on_data_input": false,
#             "clear_input_buffer_after_process": true,
#             "clear_output_buffer_after_process": true,
#             "print_buffer_size": false
#           },
#           "outputs": {}
#         }
#       }
#     }
#   }