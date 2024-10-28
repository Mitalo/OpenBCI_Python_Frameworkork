import pandas as pd

# Carregar o dataset
file_path = './out_final_italo2.csv'  # Substitua pelo caminho correto do seu arquivo
data = pd.read_csv(file_path)

# Encontrar o índice da primeira alteração no 'marker'
first_change_index = data['marker'].ne(data['marker'].iloc[0]).idxmax()

# Cortar os dados até esse ponto
data_cortada = data.iloc[first_change_index:]

# Verificar o resultado
print(data_cortada.head())

# Se quiser salvar o novo dataset cortado:
data_cortada.to_csv('out_final_italo2_cut.csv', index=False)
