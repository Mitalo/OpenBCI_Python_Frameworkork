import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Criando o DataFrame
dados = {
    "Voluntário": ["Antonio", "Angelo", "Gabriel", "Gustavo", "Mateus", "Murilo", "Messias", "Stefanye"],
    "Parte 4": [0.6667, 0.6, 0.9259, 0.88, 0.6923, 0.8846, 0.5385, 0.3846],
    "Parte 3": [0.7083, 0.44, 0.9259, 0.92, 0.6923, 0.8077, 0.7308, 0.3077],
    "Parte 2": [0.625, 0.48, 0.8519, 0.96, 0.4231, 0.8462, 0.6923, 0.4231],
    "Parte 1": [0.3333, 0.64, 0.8889, 0.6, 0.5385, 0.6923, 0.4231, 0.5385]
}

df = pd.DataFrame(dados)

df["Voluntário"] = [f"Voluntário {i+1}" for i in range(len(df))]

df_long = df.melt(id_vars=["Voluntário"], var_name="Parte", value_name="Acurácia")

plt.figure(figsize=(10, 6))
sns.boxplot(x="Voluntário", y="Acurácia", data=df_long, width=0.5, linewidth=1.5, showmeans=True, 
    meanprops={"marker": "X", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 7},
    boxprops={"facecolor": "none", "edgecolor": "black"},  # Faz as caixas ficarem vazadas
    whiskerprops={"color": "black"},  # Cor das linhas externas
    capprops={"color": "black"},  # Cor dos traços superiores e inferiores
    medianprops={"color": "black"},  # Cor da linha mediana
    flierprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 5}  # Outliers vazados
)

# Personalizando o gráfico
# plt.title("Boxplot das Porcentagens por Voluntário")
plt.ylabel("Acurácia (%)")
plt.xlabel("Voluntário")

# Ajustando os rótulos do eixo X
plt.xticks(rotation=45, ha="right")  # Rotaciona e alinha à direita
plt.ylim(0, df_long["Acurácia"].max()+0.05)  # Começa em 0
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()  # Ajusta o layout automaticamente para evitar cortes

plt.show()

