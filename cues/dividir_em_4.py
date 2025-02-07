import os
import pandas as pd

# Diretório onde estão os arquivos CSV
directory = "C:/Users/italo/OneDrive/Documentos/GitHub/OpenBCI_Python_Frameworkork/cues/coletas/treino novo/train"  # Altere para o caminho correto
# Diretório de destino dos arquivos divididos
destination_directory = "C:/Users/italo/OneDrive/Documentos/GitHub/OpenBCI_Python_Frameworkork/cues/coletas/treino novo/divididos"  # Altere para o caminho desejado
os.makedirs(destination_directory, exist_ok=True)

# Função para dividir um arquivo CSV em 4 partes
def split_csv(file_path):
    df = pd.read_csv(file_path)
    total_rows = len(df)
    
    # Definir os pontos de corte
    split_size = total_rows // 4
    splits = [df.iloc[i * split_size : (i + 1) * split_size] for i in range(3)]
    splits.append(df.iloc[3 * split_size :])  # Última parte pode ser maior
    
    # Nome base do arquivo
    file_name, file_ext = os.path.splitext(os.path.basename(file_path))
    
    # Salvar cada parte em um novo arquivo
    for i, split_df in enumerate(splits):
        new_file = os.path.join(destination_directory, f"{file_name}_part{i+1}{file_ext}")
        split_df.to_csv(new_file, index=False)
        print(f"Arquivo salvo: {new_file}")

# Percorrer todos os arquivos do diretório
for file in os.listdir(directory):
    if file.endswith(".csv") and ("_data" in file or "_events" in file):
        file_path = os.path.join(directory, file)
        print(f"Processando: {file}")
        split_csv(file_path)

print("Processamento concluído!")
