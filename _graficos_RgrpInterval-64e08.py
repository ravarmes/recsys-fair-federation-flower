import numpy as np
import matplotlib.pyplot as plt

from _resultados_64e08 import data_rgrp_activity_FedAvgExample_means, data_rgrp_activity_FedAvgExample_confidence_interval
from _resultados_64e08 import data_rgrp_activity_FedAvgLoss_means, data_rgrp_activity_FedAvgLoss_confidence_interval
from _resultados_64e08 import data_rgrp_activity_FedFairLoss_means, data_rgrp_activity_FedFairLoss_confidence_interval
from _resultados_64e08 import data_rgrp_activity_FedDEEVLoss_means, data_rgrp_activity_FedDEEVLoss_confidence_interval

rounds = list(range(0, 13))


# Criação da figura e dos subplots
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 9))

# Subplot 1
ax1.plot(rounds, data_rgrp_activity_FedAvgExample_means, label=r"FedAvg($n$)", linestyle='-')
ax1.plot(rounds, data_rgrp_activity_FedAvgLoss_means, label=r"FedAvg($\ell$)", linestyle='-')
ax1.plot(rounds, data_rgrp_activity_FedFairLoss_means, label=r"FedFair($\ell$)", linestyle='-')
# ax1.plot(rounds, data_rgrp_activity_FedDEEVLoss_means, label=r"FedDEEV($\ell$)", linestyle='-')

ax1.fill_between(rounds, data_rgrp_activity_FedAvgExample_means - data_rgrp_activity_FedAvgExample_confidence_interval, data_rgrp_activity_FedAvgExample_means + data_rgrp_activity_FedAvgExample_confidence_interval, color='b', alpha=0.2)
ax1.fill_between(rounds, data_rgrp_activity_FedAvgLoss_means - data_rgrp_activity_FedAvgLoss_confidence_interval, data_rgrp_activity_FedAvgLoss_means + data_rgrp_activity_FedAvgLoss_confidence_interval, color='orange', alpha=0.2)
ax1.fill_between(rounds, data_rgrp_activity_FedFairLoss_means - data_rgrp_activity_FedFairLoss_confidence_interval, data_rgrp_activity_FedFairLoss_means + data_rgrp_activity_FedFairLoss_confidence_interval, color='g', alpha=0.2)
# ax1.fill_between(rounds, data_rgrp_activity_FedDEEVLoss_means - data_rgrp_activity_FedDEEVLoss_confidence_interval, data_rgrp_activity_FedDEEVLoss_means + data_rgrp_activity_FedDEEVLoss_confidence_interval, color='r', alpha=0.2)

ax1.set_ylabel(r"$R_{grp}$", fontsize=14)
ax1.set_title(r"Atividade")
ax1.legend(loc='lower center')
# ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)

# # Subplot 2
# ax2.plot(data_rgrp_age_FedAvg["Round"], data_rgrp_age_FedAvg_means, label=r"FedAvg($n$)", linestyle='-')
# ax2.plot(data_rgrp_age_FedLoss["Round"], data_rgrp_age_FedLoss_means, label=r"FedAvg($\ell$)", linestyle='-')
# ax2.plot(data_rgrp_age_FairFed["Round"], data_rgrp_age_FairFed_means, label=r"FedFair($\ell$)", linestyle='-')

# ax2.fill_between(data_rgrp_age_FedAvg["Round"], data_rgrp_age_FedAvg_means - data_rgrp_age_FedAvg_confidence_interval, data_rgrp_age_FedAvg_means + data_rgrp_age_FedAvg_confidence_interval, color='b', alpha=0.2)
# ax2.fill_between(data_rgrp_age_FedLoss["Round"], data_rgrp_age_FedLoss_means - data_rgrp_age_FedLoss_confidence_interval, data_rgrp_age_FedLoss_means + data_rgrp_age_FedLoss_confidence_interval, color='r', alpha=0.2)
# ax2.fill_between(data_rgrp_age_FairFed["Round"], data_rgrp_age_FairFed_means - data_rgrp_age_FairFed_confidence_interval, data_rgrp_age_FairFed_means + data_rgrp_age_FairFed_confidence_interval, color='g', alpha=0.2)

# #ax2.set_ylabel(r"$R_{grp}$", fontsize=14)
# ax2.set_title(r"Idade")
# ax2.legend(loc='lower center')


# # Subplot 3
# ax3.plot(data_rgrp_gender_FedAvg["Round"], data_rgrp_gender_FedAvg_means, label=r"FedAvg($n$)", linestyle='-')
# ax3.plot(data_rgrp_gender_FedLoss["Round"], data_rgrp_gender_FedLoss_means, label=r"FedAvg($\ell$)", linestyle='-')
# ax3.plot(data_rgrp_gender_FairFed["Round"], data_rgrp_gender_FairFed_means, label=r"FedFair($\ell$)", linestyle='-')

# ax3.fill_between(data_rgrp_gender_FedAvg["Round"], data_rgrp_gender_FedAvg_means - data_rgrp_gender_FedAvg_confidence_interval, data_rgrp_gender_FedAvg_means + data_rgrp_gender_FedAvg_confidence_interval, color='b', alpha=0.2)
# ax3.fill_between(data_rgrp_gender_FedLoss["Round"], data_rgrp_gender_FedLoss_means - data_rgrp_gender_FedLoss_confidence_interval, data_rgrp_gender_FedLoss_means + data_rgrp_gender_FedLoss_confidence_interval, color='r', alpha=0.2)
# ax3.fill_between(data_rgrp_gender_FairFed["Round"], data_rgrp_gender_FairFed_means - data_rgrp_gender_FairFed_confidence_interval, data_rgrp_gender_FairFed_means + data_rgrp_gender_FairFed_confidence_interval, color='g', alpha=0.2)

# ax3.set_title(r"Gênero")
# ax3.legend(loc='lower center')


# # Subplot 4
# ax4.plot(data_rmse_FedAvg["Round"], data_rmse_FedAvg["RMSE"], label=r"FedAvg($n$)", linestyle='-')
# ax4.plot(data_rmse_FedLoss["Round"], data_rmse_FedLoss["RMSE"], label=r"FedAvg($\ell$)", linestyle='-')
# ax4.plot(data_rmse_activity_FairFed["Round"], data_rmse_activity_FairFed["RMSE"], label=r"FedFair($\ell$)", linestyle='-')

# ax4.set_xlabel("Round")
# ax4.set_ylabel(r"$RMSE$", fontsize=14)
# ax4.legend()

# # Subplot 5
# ax5.plot(data_rmse_FedAvg["Round"], data_rmse_FedAvg["RMSE"], label=r"FedAvg($n$)", linestyle='-')
# ax5.plot(data_rmse_FedLoss["Round"], data_rmse_FedLoss["RMSE"], label=r"FedAvg($\ell$)", linestyle='-')
# ax5.plot(data_rmse_age_FairFed["Round"], data_rmse_age_FairFed["RMSE"], label=r"FedFair($\ell$)", linestyle='-')

# ax5.set_xlabel("Round")
# ax5.legend()


# # Subplot 6
# ax6.plot(data_rmse_FedAvg["Round"], data_rmse_FedAvg["RMSE"], label=r"FedAvg($n$)", linestyle='-')
# ax6.plot(data_rmse_FedLoss["Round"], data_rmse_FedLoss["RMSE"], label=r"FedAvg($\ell$)", linestyle='-')
# ax6.plot(data_rmse_gender_FairFed["Round"], data_rmse_gender_FairFed["RMSE"], label=r"FedFair($\ell$)", linestyle='-')


# ax6.set_xlabel("Round")
# ax6.legend()

plt.subplots_adjust(hspace=0.8, wspace=0.5)

# Ajustar espaçamento entre subplots
plt.tight_layout()
plt.show()
