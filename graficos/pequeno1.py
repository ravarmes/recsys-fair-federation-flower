import matplotlib.pyplot as plt
import numpy as np

# Dados do gráfico
valores_Rgrp = [0.516497473, 0.299465100, 0.303121527, 0.303058505, 0.123925645, 0.114153242, 0.136963166]
valores_RMSE = [1.778912981, 2.278090557, 2.242979129, 2.233993689, 1.491185466, 1.441035231, 1.462143381]
categorias_labels = ['NFS', 'FedAVG', 'FedW-Rindv', 'FedW-Loss', 'FedAVG(π)', 'FedW-Rindv(π)', 'FedW-Loss(π)']

# Configurações gerais da figura
fig_width = 3  # Largura individual de cada subplot para garantir aspecto quadrado
fig_height = 3 # Altura individual de cada subplot para garantir aspecto quadrado
num_plots = 5
total_width = fig_width * num_plots  # Largura total da figura baseado no número de plots
plt.figure(figsize=(total_width, fig_height))  # Ajuste do tamanho geral da figura

# Criando os subplots
for idx in range(5):  # Loop para criar 5 gráficos
    ax1 = plt.subplot(1, 5, idx+1)
    ax1.set_facecolor('#d9d9d9')  # Definindo a cor de fundo para cada gráfico como cinza
    ax1.plot(categorias_labels, valores_Rgrp, marker='o', linestyle='-', color='dodgerblue', label='Rgrp', linewidth=2)
    ax2 = ax1.twinx()
    ax2.plot(categorias_labels, valores_RMSE, marker='s', linestyle='--', color='crimson', label='RMSE', linewidth=2)

    # Ajustando as configurações estéticas
    ax1.set_ylabel('Injustiça do Grupo (Rgrp)' if idx == 0 else '', color='dodgerblue')
    ax2.set_ylabel('Erro Quadrático Médio (RMSE)' if idx == 4 else '', color='crimson')
    ax1.tick_params(labelsize=8)
    ax2.tick_params(labelsize=8)

    # Legenda, apenas uma vez
    # if idx == 0:
    ax1.legend(loc='upper right', fontsize='small')
    ax2.legend(loc='upper left', fontsize='small')

    # Definindo limites para sincronizar os eixos entre os gráficos
    ax1.set_ylim(0, 0.6)
    ax2.set_ylim(1, 3)

    # Rotação dos rótulos para evitar sobreposição
    plt.xticks(rotation=45, fontsize=8)
    # Linhas de grade brancas
    ax1.grid(True, which='both', linestyle='-', linewidth='0.5', color='white', alpha=0.7)

    # Removendo bordas dos eixos
    for spine in ax1.spines.values():
        spine.set_visible(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)

plt.subplots_adjust(wspace=0.1)  # Reduz o espaço entre os gráficos
plt.tight_layout()
plt.show()
