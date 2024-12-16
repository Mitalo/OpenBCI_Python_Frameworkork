import pandas as pd
import os

def process_csv_file(input_file_path, output_file_path):
    """
    Carrega um arquivo CSV, filtra as linhas onde o marker é igual a 0 e salva o arquivo em outra pasta.

    :param input_file_path: Caminho do arquivo CSV de entrada.
    :param output_file_path: Caminho do arquivo CSV de saída.
    """
    # Carregar o arquivo CSV
    df = pd.read_csv(input_file_path)

    # Filtrar linhas onde o marker é igual a 0
    df_filtered = df[df['marker'] != 0]

    # Salvar o arquivo CSV processado
    df_filtered.to_csv(output_file_path, index=False)
    print(f"Arquivo processado e salvo: {output_file_path}")

def process_multiple_csv_files(input_directory, output_directory):
    """
    Processa múltiplos arquivos CSV em um diretório, filtrando as linhas onde o marker é igual a 0 e salvando os arquivos em outra pasta.

    :param input_directory: Diretório contendo os arquivos CSV de entrada.
    :param output_directory: Diretório onde os arquivos CSV processados serão salvos.
    """
    # Criar o diretório de saída se não existir
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Percorrer todos os arquivos no diretório de entrada
    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            input_file_path = os.path.join(input_directory, filename)
            output_file_path = os.path.join(output_directory, filename)
            process_csv_file(input_file_path, output_file_path)

# Exemplo de uso
input_directory = 'C:/Users/italo/OneDrive/Documentos/GitHub/OpenBCI_Python_Frameworkork/cues/coletas/treino novo'  # Substitua pelo caminho da pasta de entrada
output_directory = 'C:/Users/italo/OneDrive/Documentos/GitHub/OpenBCI_Python_Frameworkork/cues/coletas/treino novo'  # Substitua pelo caminho da pasta de saída

# Processar arquivos CSV
process_multiple_csv_files(input_directory, output_directory)