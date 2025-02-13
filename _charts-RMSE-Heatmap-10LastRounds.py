import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Importar dados
from _results_MovieLens_RMSE_26e14 import data_rmse_activity_FedAvgExample_means, data_rmse_activity_FedAvgLoss_means, data_rmse_activity_FedFairLoss_means
from _results_MovieLens_RMSE_26e14 import data_rmse_age_FedAvgExample_means, data_rmse_age_FedAvgLoss_means, data_rmse_age_FedFairLoss_means
from _results_MovieLens_RMSE_26e14 import data_rmse_gender_FedAvgExample_means, data_rmse_gender_FedAvgLoss_means, data_rmse_gender_FedFairLoss_means

from _results_GoodBooks_RMSE_05e02 import data_rmse_activity_FedAvgExample_means_goodbooks, data_rmse_activity_FedAvgLoss_means_goodbooks, data_rmse_activity_FedFairLoss_means_goodbooks

# Calcular as médias dos últimos 10 rounds
def calc_media_ultimos_10(dados):
    return np.mean(dados[-10:])

rgrp_activity_FedAvg_n = calc_media_ultimos_10(data_rmse_activity_FedAvgExample_means)
rgrp_age_FedAvg_n = calc_media_ultimos_10(data_rmse_activity_FedAvgExample_means)
rgrp_gender_FedAvg_n = calc_media_ultimos_10(data_rmse_activity_FedAvgExample_means)

rgrp_activity_FedAvg_l = calc_media_ultimos_10(data_rmse_activity_FedAvgLoss_means)
rgrp_age_FedAvg_l = calc_media_ultimos_10(data_rmse_activity_FedAvgLoss_means)
rgrp_gender_FedAvg_l = calc_media_ultimos_10(data_rmse_activity_FedAvgLoss_means)

rgrp_activity_FairFed_l = calc_media_ultimos_10(data_rmse_activity_FedFairLoss_means)
rgrp_age_FairFed_l = calc_media_ultimos_10(data_rmse_age_FedFairLoss_means)
rgrp_gender_FairFed_l = calc_media_ultimos_10(data_rmse_gender_FedFairLoss_means)

rgrp_activity_FedAvg_n_goodbooks = calc_media_ultimos_10(data_rmse_activity_FedAvgExample_means_goodbooks)
rgrp_activity_FedAvg_l_goodbooks = calc_media_ultimos_10(data_rmse_activity_FedAvgLoss_means_goodbooks)
rgrp_activity_FairFed_l_goodbooks = calc_media_ultimos_10(data_rmse_activity_FedFairLoss_means_goodbooks)


# Cálculo do aumento percentual com ajuste
aumento_activity_FedAvg_n = 100 * (rgrp_activity_FairFed_l - rgrp_activity_FedAvg_n) / max(rgrp_activity_FedAvg_n, 1e-10)
aumento_activity_FedAvg_l = 100 * (rgrp_activity_FairFed_l - rgrp_activity_FedAvg_l) / max(rgrp_activity_FedAvg_l, 1e-10)

aumento_age_FedAvg_n = 100 * (rgrp_age_FairFed_l - rgrp_age_FedAvg_n) / max(rgrp_age_FedAvg_n, 1e-10)
aumento_age_FedAvg_l = 100 * (rgrp_age_FairFed_l - rgrp_age_FedAvg_l) / max(rgrp_age_FedAvg_l, 1e-10)

aumento_gender_FedAvg_n = 100 * (rgrp_gender_FairFed_l - rgrp_gender_FedAvg_n) / max(rgrp_gender_FedAvg_n, 1e-10)
aumento_gender_FedAvg_l = 100 * (rgrp_gender_FairFed_l - rgrp_gender_FedAvg_l) / max(rgrp_gender_FedAvg_l, 1e-10)

aumento_activity_FedAvg_n_goodbooks = 100 * (rgrp_activity_FairFed_l_goodbooks - rgrp_activity_FedAvg_n_goodbooks) / max(rgrp_activity_FedAvg_n_goodbooks, 1e-10)
aumento_activity_FedAvg_l_goodbooks = 100 * (rgrp_activity_FairFed_l - rgrp_activity_FedAvg_l_goodbooks) / max(rgrp_activity_FedAvg_l_goodbooks, 1e-10)

# Garantir que o aumento seja não negativo
aumento_activity_FedAvg_n = max(aumento_activity_FedAvg_n, 0)
aumento_activity_FedAvg_l = max(aumento_activity_FedAvg_l, 0)

aumento_age_FedAvg_n = max(aumento_age_FedAvg_n, 0)
aumento_age_FedAvg_l = max(aumento_age_FedAvg_l, 0)

aumento_gender_FedAvg_n = max(aumento_gender_FedAvg_n, 0)
aumento_gender_FedAvg_l = max(aumento_gender_FedAvg_l, 0)

aumento_activity_FedAvg_n_goodbooks = max(aumento_activity_FedAvg_n_goodbooks, 0)
aumento_activity_FedAvg_l_goodbooks = max(aumento_activity_FedAvg_l_goodbooks, 0)

# Matriz de aumentos
aumento_matrix = np.array([
    [aumento_activity_FedAvg_n, aumento_age_FedAvg_n, aumento_gender_FedAvg_n, aumento_activity_FedAvg_n_goodbooks],
    [aumento_activity_FedAvg_l, aumento_age_FedAvg_l, aumento_gender_FedAvg_l, aumento_activity_FedAvg_l_goodbooks]
])

# Configuração do heatmap com tons de verde mais suave
cmap = mcolors.LinearSegmentedColormap.from_list("lighter_blues", ["#cce5ff", "#66b3ff", "#3385ff"])

algorithms = [r"FedAvg($n$)", r"FedAvg($\ell$)"]
categories = ["MovieLens\nActivity", "MovieLens\nAge", "MovieLens\nGender", "GoodBooks\nActivity"]

plt.figure(figsize=(8, 5))
heatmap = plt.imshow(aumento_matrix, cmap=cmap, aspect="auto")

# Adicionar rótulos
plt.xticks(np.arange(len(categories)), categories, fontsize=12)
plt.yticks(np.arange(len(algorithms)), algorithms, fontsize=12)

# Adicionar valores nas células
for i in range(len(algorithms)):
    for j in range(len(categories)):
        plt.text(j, i, f"{aumento_matrix[i, j]:.2f}%", ha="center", va="center", color="black", fontsize=12)

# Adicionar barra de cores
plt.colorbar(heatmap, label="Increase (%)")

plt.title("Percentage Increase in $RMSE$ Compared to FedFair($\ell$) - Last 10 Rounds")
plt.tight_layout()
plt.show()
