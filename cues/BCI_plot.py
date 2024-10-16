import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados
df = pd.read_csv('./out_final.csv')

# Normalizar o 'marker' para ser um sinal quadrado
df['marker_normalizado'] = (df['marker'] - df['marker'].min()) / (df['marker'].max() - df['marker'].min())

# Criar a figura
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotar os sinais Fp1 e C3 no eixo principal
ax1.plot(df['time_board'], df['Fp1'], label='Fp1', color='blue')
ax1.plot(df['time_board'], df['C3'], label='C3', color='green')

# Configurações para o eixo principal
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude dos Sinais', color='black')
ax1.legend(loc='upper left')
ax1.tick_params(axis='y')

# Criar um segundo eixo Y para o marker
ax2 = ax1.twinx()

# Plotar o marker no eixo secundário
ax2.step(df['time_board'], df['marker_normalizado'], label='Marker', color='red', where='post')

# Configurações para o eixo secundário
ax2.set_ylabel('Marker (Sinal Quadrado)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Adicionar a legenda do marker no eixo secundário
ax2.legend(loc='upper right')

# Mostrar o gráfico
plt.title('Sinais Fp1, C3 e Marker (com eixo secundário)')
plt.tight_layout()
plt.show()
