import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Importar dados
from _results_MovieLens_Rgrp_26e14 import data_rgrp_activity_FedAvgExample_means, data_rgrp_activity_FedAvgLoss_means, data_rgrp_activity_FedFairLoss_means
from _results_MovieLens_Rgrp_26e14 import data_rgrp_age_FedAvgExample_means, data_rgrp_age_FedAvgLoss_means, data_rgrp_age_FedFairLoss_means
from _results_MovieLens_Rgrp_26e14 import data_rgrp_gender_FedAvgExample_means, data_rgrp_gender_FedAvgLoss_means, data_rgrp_gender_FedFairLoss_means

from _results_GoodBooks_Rgrp_05e02 import data_rgrp_activity_FedAvgExample_means_goodbooks, data_rgrp_activity_FedAvgLoss_means_goodbooks, data_rgrp_activity_FedFairLoss_means_goodbooks


# Calcular as médias dos últimos 10 rounds
def calc_media_ultimos_10(dados):
    return np.mean(dados[-10:])

rgrp_activity_FedAvg_n = calc_media_ultimos_10(data_rgrp_activity_FedAvgExample_means)
rgrp_age_FedAvg_n = calc_media_ultimos_10(data_rgrp_age_FedAvgExample_means)
rgrp_gender_FedAvg_n = calc_media_ultimos_10(data_rgrp_gender_FedAvgExample_means)

rgrp_activity_FedAvg_l = calc_media_ultimos_10(data_rgrp_activity_FedAvgLoss_means)
rgrp_age_FedAvg_l = calc_media_ultimos_10(data_rgrp_age_FedAvgLoss_means)
rgrp_gender_FedAvg_l = calc_media_ultimos_10(data_rgrp_gender_FedAvgLoss_means)

rgrp_activity_FairFed_l = calc_media_ultimos_10(data_rgrp_activity_FedFairLoss_means)
rgrp_age_FairFed_l = calc_media_ultimos_10(data_rgrp_age_FedFairLoss_means)
rgrp_gender_FairFed_l = calc_media_ultimos_10(data_rgrp_gender_FedFairLoss_means)

rgrp_activity_FedAvg_n_goodbooks = calc_media_ultimos_10(data_rgrp_activity_FedAvgExample_means_goodbooks)
rgrp_activity_FedAvg_l_goodbooks = calc_media_ultimos_10(data_rgrp_activity_FedAvgLoss_means_goodbooks)
rgrp_activity_FairFed_l_goodbooks = calc_media_ultimos_10(data_rgrp_activity_FedFairLoss_means_goodbooks)


# Cálculo da redução percentual com ajuste para não ultrapassar 100%
reducao_activity_FedAvg_n = 100 * (rgrp_activity_FedAvg_n - rgrp_activity_FairFed_l) / max(rgrp_activity_FedAvg_n, 1e-10)
reducao_activity_FedAvg_l = 100 * (rgrp_activity_FedAvg_l - rgrp_activity_FairFed_l) / max(rgrp_activity_FedAvg_l, 1e-10)

reducao_age_FedAvg_n = 100 * (rgrp_age_FedAvg_n - rgrp_age_FairFed_l) / max(rgrp_age_FedAvg_n, 1e-10)
reducao_age_FedAvg_l = 100 * (rgrp_age_FedAvg_l - rgrp_age_FairFed_l) / max(rgrp_age_FedAvg_l, 1e-10)

reducao_gender_FedAvg_n = 100 * (rgrp_gender_FedAvg_n - rgrp_gender_FairFed_l) / max(rgrp_gender_FedAvg_n, 1e-10)
reducao_gender_FedAvg_l = 100 * (rgrp_gender_FedAvg_l - rgrp_gender_FairFed_l) / max(rgrp_gender_FedAvg_l, 1e-10)

reducao_activity_FedAvg_n_goodbooks = 100 * (rgrp_activity_FedAvg_n_goodbooks - rgrp_activity_FairFed_l_goodbooks) / max(rgrp_activity_FedAvg_n_goodbooks, 1e-10)
reducao_activity_FedAvg_l_goodbooks = 100 * (rgrp_activity_FedAvg_l_goodbooks - rgrp_activity_FairFed_l_goodbooks) / max(rgrp_activity_FedAvg_l_goodbooks, 1e-10)

# Garantir que a redução não ultrapasse 100%
reducao_activity_FedAvg_n = min(reducao_activity_FedAvg_n, 100)
reducao_activity_FedAvg_l = min(reducao_activity_FedAvg_l, 100)

reducao_age_FedAvg_n = min(reducao_age_FedAvg_n, 100)
reducao_age_FedAvg_l = min(reducao_age_FedAvg_l, 100)

reducao_gender_FedAvg_n = min(reducao_gender_FedAvg_n, 100)
reducao_gender_FedAvg_l = min(reducao_gender_FedAvg_l, 100)

reducao_activity_FedAvg_n_goodbooks = min(reducao_activity_FedAvg_n_goodbooks, 100)
reducao_activity_FedAvg_l_goodbooks = min(reducao_activity_FedAvg_l_goodbooks, 100)

# Matriz de reduções
reducao_matrix = np.array([
    [reducao_activity_FedAvg_n, reducao_age_FedAvg_n, reducao_gender_FedAvg_n, reducao_activity_FedAvg_n_goodbooks],
    [reducao_activity_FedAvg_l, reducao_age_FedAvg_l, reducao_gender_FedAvg_l, reducao_activity_FedAvg_l_goodbooks]
])

# Configuração do heatmap com tons de vermelho mais suave
cmap = mcolors.LinearSegmentedColormap.from_list("lighter_reds", ["#ffcccc", "#ff6666", "#ff3333"])

algorithms = [r"FedAvg($n$)", r"FedAvg($\ell$)"]
categories = ["MovieLens\nActivity", "MovieLens\nAge", "MovieLens\nGender", "GoodBooks\nActivity"]

plt.figure(figsize=(8, 5))
heatmap = plt.imshow(reducao_matrix, cmap=cmap, aspect="auto")

# Adicionar rótulos
plt.xticks(np.arange(len(categories)), categories, fontsize=12)
plt.yticks(np.arange(len(algorithms)), algorithms, fontsize=12)

# Adicionar valores nas células
for i in range(len(algorithms)):
    for j in range(len(categories)):
        plt.text(j, i, f"{reducao_matrix[i, j]:.2f}%", ha="center", va="center", color="black", fontsize=12)

# Adicionar barra de cores
plt.colorbar(heatmap, label="Reduction (%)")

plt.title("Percentage Reduction of $R_{grp}$ Compared to FedFair($\ell$) - Last 10 Rounds")
plt.tight_layout()
plt.show()
