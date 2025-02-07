# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import glob
# from sklearn.metrics import confusion_matrix

# # 📂 Defina o diretório onde estão os arquivos CSV
# pasta_dados = "C:/Users/italo/OneDrive/Documentos/TCC/Saídas teste/"  # Substituir pelo caminho real

# # 🏷️ Lista de voluntários (adicione ou modifique conforme necessário)
# voluntarios = ["antonio", "angelo", "gabriel", "gustavo", "mateus", "murilo", "messias", "stefanye"]

# # 🏷️ Labels das classes (modifique conforme necessário)
# labels = [1, 2, 3, 4, 5]
# labels_map = {1: "A", 2: "E", 3: "I", 4: "O", 5: "U"}


# # 🔄 Processar cada voluntário separadamente
# for i, voluntario in enumerate(voluntarios):
#     matriz_compilada = np.zeros((len(labels), len(labels)))  # Inicializa a matriz acumulada

#     # 🔄 Percorrer as partes (1 a 4)
#     for parte in range(1, 5):
#         arquivo_real = f"{pasta_dados}real_{voluntario}_part{parte}.csv"
#         arquivo_prev = f"{pasta_dados}previsoes_{voluntario}_part{parte}.csv"

#         # 📥 Carregar os dados dos arquivos CSV
#         real = pd.read_csv(arquivo_real).values.ravel()  # Transformar em vetor 1D
#         previsoes = pd.read_csv(arquivo_prev).values.ravel()  # Transformar em vetor 1D

#         # ⚡ Gerar a matriz de confusão para esta parte
#         matriz = confusion_matrix(real, previsoes, labels=labels)

#         # ➕ Somar à matriz compilada
#         matriz_compilada += matriz

#     matriz_percentual = matriz_compilada / matriz_compilada.sum(axis=1, keepdims=True)

#     # 🔥 Criar o heatmap da matriz compilada
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(
#         matriz_percentual,
#         annot=True,
#         fmt=".2%",
#         cmap="Blues",
#         linewidths=0.5,
#         xticklabels=[labels_map[l] for l in labels],  # Substitui números por letras
#         yticklabels=[labels_map[l] for l in labels],
#         cbar=False
#     )

#     # 🎨 Personalizações do gráfico
#     # plt.title(f"Matriz de Confusão - Voluntário {i+1}")
#     plt.xlabel("Previsões")
#     plt.ylabel("Valores Reais")

#     # 📤 Mostrar o gráfico
#     plt.show()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import confusion_matrix

# 📂 Defina o diretório onde estão os arquivos CSV
pasta_dados = "C:/Users/italo/OneDrive/Documentos/TCC/Saídas teste/"  # Substituir pelo caminho real

# 🏷️ Lista de voluntários (adicione ou modifique conforme necessário)
voluntarios = ["antonio", "angelo", "gabriel", "gustavo", "mateus", "murilo", "messias", "stefanye"]

# 🏷️ Labels das classes (modifique conforme necessário)
labels = [1, 2, 3, 4, 5]
labels_map = {1: "A", 2: "E", 3: "I", 4: "O", 5: "U"}

# Inicializa a matriz compilada para todos os voluntários
matriz_compilada = np.zeros((len(labels), len(labels)))  # Inicializa a matriz acumulada

# 🔄 Processar cada voluntário separadamente
for voluntario in voluntarios:
    # 🔄 Percorrer as partes (1 a 4) para cada voluntário
    for parte in range(1, 5):
        arquivo_real = f"{pasta_dados}real_{voluntario}_part{parte}.csv"
        arquivo_prev = f"{pasta_dados}previsoes_{voluntario}_part{parte}.csv"

        # 📥 Carregar os dados dos arquivos CSV
        real = pd.read_csv(arquivo_real).values.ravel()  # Transformar em vetor 1D
        previsoes = pd.read_csv(arquivo_prev).values.ravel()  # Transformar em vetor 1D

        # ⚡ Gerar a matriz de confusão para esta parte
        matriz = confusion_matrix(real, previsoes, labels=labels)

        # ➕ Somar à matriz compilada
        matriz_compilada += matriz

# Converter a matriz compilada para percentual
matriz_percentual = matriz_compilada / matriz_compilada.sum(axis=1, keepdims=True)

# 🔥 Criar o heatmap da matriz compilada
plt.figure(figsize=(6, 5))
sns.heatmap(
    matriz_percentual,
    annot=True,
    fmt=".2%",
    cmap="Blues",
    linewidths=0.5,
    xticklabels=[labels_map[l] for l in labels],  # Substitui números por letras
    yticklabels=[labels_map[l] for l in labels],
    cbar=False
)

# 🎨 Personalizações do gráfico
plt.xlabel("Previsões")
plt.ylabel("Valores Reais")

# 📤 Mostrar o gráfico
plt.show()
