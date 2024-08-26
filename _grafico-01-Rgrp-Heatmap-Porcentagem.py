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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Dados finais (último round)
rgrp_activity_FedAvg_n = data_rgrp_activity_FedAvg["RgrpActivity"][-1]
rgrp_age_FedAvg_n = data_rgrp_age_FedAvg["RgrpAge"][-1]
rgrp_gender_FedAvg_n = data_rgrp_gender_FedAvg["RgrpGender"][-1]

rgrp_activity_FedAvg_l = data_rgrp_activity_FedLoss["RgrpActivity"][-1]
rgrp_age_FedAvg_l = data_rgrp_age_FedLoss["RgrpAge"][-1]
rgrp_gender_FedAvg_l = data_rgrp_gender_FedLoss["RgrpGender"][-1]

rgrp_activity_FairFed_l = data_rgrp_activity_FairFed["RgrpActivity"][-1]
rgrp_age_FairFed_l = data_rgrp_age_FairFed["RgrpAge"][-1]
rgrp_gender_FairFed_l = data_rgrp_gender_FairFed["RgrpGender"][-1]

# Cálculo da redução percentual com ajuste para não ultrapassar 100%
reducao_activity_FedAvg_n = 100 * (rgrp_activity_FedAvg_n - rgrp_activity_FairFed_l) / max(rgrp_activity_FedAvg_n, 1e-10)
reducao_activity_FedAvg_l = 100 * (rgrp_activity_FedAvg_l - rgrp_activity_FairFed_l) / max(rgrp_activity_FedAvg_l, 1e-10)

reducao_age_FedAvg_n = 100 * (rgrp_age_FedAvg_n - rgrp_age_FairFed_l) / max(rgrp_age_FedAvg_n, 1e-10)
reducao_age_FedAvg_l = 100 * (rgrp_age_FedAvg_l - rgrp_age_FairFed_l) / max(rgrp_age_FedAvg_l, 1e-10)

reducao_gender_FedAvg_n = 100 * (rgrp_gender_FedAvg_n - rgrp_gender_FairFed_l) / max(rgrp_gender_FedAvg_n, 1e-10)
reducao_gender_FedAvg_l = 100 * (rgrp_gender_FedAvg_l - rgrp_gender_FairFed_l) / max(rgrp_gender_FedAvg_l, 1e-10)

# Garantir que a redução não ultrapasse 100%
reducao_activity_FedAvg_n = min(reducao_activity_FedAvg_n, 100)
reducao_activity_FedAvg_l = min(reducao_activity_FedAvg_l, 100)

reducao_age_FedAvg_n = min(reducao_age_FedAvg_n, 100)
reducao_age_FedAvg_l = min(reducao_age_FedAvg_l, 100)

reducao_gender_FedAvg_n = min(reducao_gender_FedAvg_n, 100)
reducao_gender_FedAvg_l = min(reducao_gender_FedAvg_l, 100)

# Matriz de reduções
reducao_matrix = np.array([
    [reducao_activity_FedAvg_n, reducao_age_FedAvg_n, reducao_gender_FedAvg_n],
    [reducao_activity_FedAvg_l, reducao_age_FedAvg_l, reducao_gender_FedAvg_l]
])

# Configuração do heatmap com tons de vermelho mais suave
cmap = mcolors.LinearSegmentedColormap.from_list("lighter_reds", ["#ffcccc", "#ff6666", "#ff3333"])

algorithms = [r"FedAvg($n$)", r"FedAvg($\ell$)"]
categories = ["Atividade", "Idade", "Gênero"]

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
plt.colorbar(heatmap, label="Redução (%)")

plt.title("Redução Percentual de $R_{grp}$ Comparando com FedFair($\ell$)")
plt.tight_layout()
plt.show()
