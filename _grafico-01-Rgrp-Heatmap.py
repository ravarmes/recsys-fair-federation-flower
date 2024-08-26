import matplotlib.pyplot as plt
import numpy as np

from _resultados_FedAvgExample import data_rgrp_activity_FedAvg
from _resultados_FedAvgExample import data_rgrp_age_FedAvg
from _resultados_FedAvgExample import data_rgrp_gender_FedAvg

from _resultados_FedAvgLoss import data_rgrp_activity_FedLoss
from _resultados_FedAvgLoss import data_rgrp_age_FedLoss
from _resultados_FedAvgLoss import data_rgrp_gender_FedLoss

from _resultados_FedFair import data_rgrp_activity_FairFed
from _resultados_FedFair import data_rgrp_age_FairFed
from _resultados_FedFair import data_rgrp_gender_FairFed

# Extrair os valores de Rgrp do último round para cada categoria
rgrp_activity_fedavg = data_rgrp_activity_FedAvg["RgrpActivity"][-1]
rgrp_activity_fedloss = data_rgrp_activity_FedLoss["RgrpActivity"][-1]
rgrp_activity_fairfed = data_rgrp_activity_FairFed["RgrpActivity"][-1]

rgrp_age_fedavg = data_rgrp_age_FedAvg["RgrpAge"][-1]
rgrp_age_fedloss = data_rgrp_age_FedLoss["RgrpAge"][-1]
rgrp_age_fairfed = data_rgrp_age_FairFed["RgrpAge"][-1]

rgrp_gender_fedavg = data_rgrp_gender_FedAvg["RgrpGender"][-1]
rgrp_gender_fedloss = data_rgrp_gender_FedLoss["RgrpGender"][-1]
rgrp_gender_fairfed = data_rgrp_gender_FairFed["RgrpGender"][-1]

# Organizar os dados para o heatmap
data = np.array([
    [rgrp_activity_fedavg, rgrp_age_fedavg, rgrp_gender_fedavg],
    [rgrp_activity_fedloss, rgrp_age_fedloss, rgrp_gender_fedloss],
    [rgrp_activity_fairfed, rgrp_age_fairfed, rgrp_gender_fairfed]
])

# Rótulos para o heatmap
algorithms = [r"FedAvg($n$)", r"FedAvg($\ell$)", r"FedFair($\ell$)"]
categories = ["Atividade", "Idade", "Gênero"]

# Criação do heatmap
fig, ax = plt.subplots(figsize=(8, 6))
heatmap = ax.imshow(data, cmap="coolwarm", aspect="auto")

# Adicionar rótulos
ax.set_xticks(np.arange(len(categories)))
ax.set_yticks(np.arange(len(algorithms)))

ax.set_xticklabels(categories)
ax.set_yticklabels(algorithms)

# Adicionar os valores nas células com cinco casas decimais
for i in range(len(algorithms)):
    for j in range(len(categories)):
        text = ax.text(j, i, f"{data[i, j]:.5f}", ha="center", va="center", color="black")

# Adicionar título e rótulos aos eixos
ax.set_title(r"$R_{grp}$ por Algoritmos em diferentes Categorias")
ax.set_xlabel("Categorias")
ax.set_ylabel("Algoritmos")

# Mostrar a barra de cores
fig.colorbar(heatmap)

plt.tight_layout()
plt.show()
