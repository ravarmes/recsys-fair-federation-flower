import numpy as np
import matplotlib.pyplot as plt

from _resultados_16e08 import data_rgrp_activity_FedAvgExample_means, data_rgrp_activity_FedAvgExample_confidence_interval
from _resultados_16e08 import data_rgrp_activity_FedAvgLoss_means, data_rgrp_activity_FedAvgLoss_confidence_interval
from _resultados_16e08 import data_rgrp_activity_FedFairLoss_means, data_rgrp_activity_FedFairLoss_confidence_interval
from _resultados_16e08 import data_rgrp_activity_FedDEEVLoss_means, data_rgrp_activity_FedDEEVLoss_confidence_interval

from _resultados_16e08 import data_rgrp_age_FedAvgExample_means, data_rgrp_age_FedAvgExample_confidence_interval
from _resultados_16e08 import data_rgrp_age_FedAvgLoss_means, data_rgrp_age_FedAvgLoss_confidence_interval
from _resultados_16e08 import data_rgrp_age_FedFairLoss_means, data_rgrp_age_FedFairLoss_confidence_interval
from _resultados_16e08 import data_rgrp_age_FedDEEVLoss_means, data_rgrp_age_FedDEEVLoss_confidence_interval

from _resultados_16e08 import data_rgrp_gender_FedAvgExample_means, data_rgrp_gender_FedAvgExample_confidence_interval
from _resultados_16e08 import data_rgrp_gender_FedAvgLoss_means, data_rgrp_gender_FedAvgLoss_confidence_interval
from _resultados_16e08 import data_rgrp_gender_FedFairLoss_means, data_rgrp_gender_FedFairLoss_confidence_interval
from _resultados_16e08 import data_rgrp_gender_FedDEEVLoss_means, data_rgrp_gender_FedDEEVLoss_confidence_interval

from _resultados_16e08_RMSE import data_rmse_activity_FedAvgExample_means, data_rmse_activity_FedAvgExample_confidence_interval
from _resultados_16e08_RMSE import data_rmse_activity_FedAvgLoss_means, data_rmse_activity_FedAvgLoss_confidence_interval
from _resultados_16e08_RMSE import data_rmse_activity_FedFairLoss_means, data_rmse_activity_FedFairLoss_confidence_interval
from _resultados_16e08_RMSE import data_rmse_age_FedFairLoss_means, data_rmse_age_FedFairLoss_confidence_interval
from _resultados_16e08_RMSE import data_rmse_gender_FedFairLoss_means, data_rmse_gender_FedFairLoss_confidence_interval

rounds = list(range(0, 49))


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
# ax1.legend(loc='lower center')
ax1.legend(loc='lower center', bbox_to_anchor=(0.35, 0), bbox_transform=ax1.transAxes)

# Subplot 2
ax2.plot(rounds, data_rgrp_age_FedAvgExample_means, label=r"FedAvg($n$)", linestyle='-')
ax2.plot(rounds, data_rgrp_age_FedAvgLoss_means, label=r"FedAvg($\ell$)", linestyle='-')
ax2.plot(rounds, data_rgrp_age_FedFairLoss_means, label=r"FedFair($\ell$)", linestyle='-')
# ax2.plot(rounds, data_rgrp_age_FedDEEVLoss_means, label=r"FedDEEV($\ell$)", linestyle='-')

ax2.fill_between(rounds, data_rgrp_age_FedAvgExample_means - data_rgrp_age_FedAvgExample_confidence_interval, data_rgrp_age_FedAvgExample_means + data_rgrp_age_FedAvgExample_confidence_interval, color='b', alpha=0.2)
ax2.fill_between(rounds, data_rgrp_age_FedAvgLoss_means - data_rgrp_age_FedAvgLoss_confidence_interval, data_rgrp_age_FedAvgLoss_means + data_rgrp_age_FedAvgLoss_confidence_interval, color='orange', alpha=0.2)
ax2.fill_between(rounds, data_rgrp_age_FedFairLoss_means - data_rgrp_age_FedFairLoss_confidence_interval, data_rgrp_age_FedFairLoss_means + data_rgrp_age_FedFairLoss_confidence_interval, color='g', alpha=0.2)
# ax2.fill_between(rounds, data_rgrp_age_FedDEEVLoss_means - data_rgrp_age_FedDEEVLoss_confidence_interval, data_rgrp_age_FedDEEVLoss_means + data_rgrp_age_FedDEEVLoss_confidence_interval, color='r', alpha=0.2)

#ax2.set_ylabel(r"$R_{grp}$", fontsize=14)
ax2.set_title(r"Idade")
# ax2.legend(loc='lower center')
ax2.legend(loc='lower center', bbox_to_anchor=(0.4, 0), bbox_transform=ax2.transAxes)


# Subplot 3
ax3.plot(rounds, data_rgrp_gender_FedAvgExample_means, label=r"FedAvg($n$)", linestyle='-')
ax3.plot(rounds, data_rgrp_gender_FedAvgLoss_means, label=r"FedAvg($\ell$)", linestyle='-')
ax3.plot(rounds, data_rgrp_gender_FedFairLoss_means, label=r"FedFair($\ell$)", linestyle='-')
# ax3.plot(rounds, data_rgrp_gender_FedDEEVLoss_means, label=r"FedDEEV($\ell$)", linestyle='-')

ax3.fill_between(rounds, data_rgrp_gender_FedAvgExample_means - data_rgrp_gender_FedAvgExample_confidence_interval, data_rgrp_gender_FedAvgExample_means + data_rgrp_gender_FedAvgExample_confidence_interval, color='b', alpha=0.2)
ax3.fill_between(rounds, data_rgrp_gender_FedAvgLoss_means - data_rgrp_gender_FedAvgLoss_confidence_interval, data_rgrp_gender_FedAvgLoss_means + data_rgrp_gender_FedAvgLoss_confidence_interval, color='orange', alpha=0.2)
ax3.fill_between(rounds, data_rgrp_gender_FedFairLoss_means - data_rgrp_gender_FedFairLoss_confidence_interval, data_rgrp_gender_FedFairLoss_means + data_rgrp_gender_FedFairLoss_confidence_interval, color='g', alpha=0.2)
# ax3.fill_between(rounds, data_rgrp_gender_FedDEEVLoss_means - data_rgrp_gender_FedDEEVLoss_confidence_interval, data_rgrp_gender_FedDEEVLoss_means + data_rgrp_gender_FedDEEVLoss_confidence_interval, color='r', alpha=0.2)

ax3.set_title(r"Gênero")
ax3.legend(loc='lower center')


# Subplot 4
ax4.plot(rounds, data_rmse_activity_FedAvgExample_means, label=r"FedAvg($n$)", linestyle='-')
ax4.plot(rounds, data_rmse_activity_FedAvgLoss_means, label=r"FedAvg($\ell$)", linestyle='-')
ax4.plot(rounds, data_rmse_activity_FedFairLoss_means, label=r"FedFair($\ell$)", linestyle='-')

ax4.fill_between(rounds, data_rmse_activity_FedAvgExample_means - data_rmse_activity_FedAvgExample_confidence_interval, data_rmse_activity_FedAvgExample_means + data_rmse_activity_FedAvgExample_confidence_interval, color='b', alpha=0.2)
ax4.fill_between(rounds, data_rmse_activity_FedAvgLoss_means - data_rmse_activity_FedAvgLoss_confidence_interval, data_rmse_activity_FedAvgLoss_means + data_rmse_activity_FedAvgLoss_confidence_interval, color='orange', alpha=0.2)
ax4.fill_between(rounds, data_rmse_activity_FedFairLoss_means - data_rmse_activity_FedFairLoss_confidence_interval, data_rmse_activity_FedFairLoss_means + data_rmse_activity_FedFairLoss_confidence_interval, color='g', alpha=0.2)

ax4.set_xlabel("Round")
ax4.set_ylabel(r"$RMSE$", fontsize=14)
ax4.legend()

# Subplot 5
ax5.plot(rounds, data_rmse_activity_FedAvgExample_means, label=r"FedAvg($n$)", linestyle='-')
ax5.plot(rounds, data_rmse_activity_FedAvgLoss_means, label=r"FedAvg($\ell$)", linestyle='-')
ax5.plot(rounds, data_rmse_age_FedFairLoss_means, label=r"FedFair($\ell$)", linestyle='-')

ax5.fill_between(rounds, data_rmse_activity_FedAvgExample_means - data_rmse_activity_FedAvgExample_confidence_interval, data_rmse_activity_FedAvgExample_means + data_rmse_activity_FedAvgExample_confidence_interval, color='b', alpha=0.2)
ax5.fill_between(rounds, data_rmse_activity_FedAvgLoss_means - data_rmse_activity_FedAvgLoss_confidence_interval, data_rmse_activity_FedAvgLoss_means + data_rmse_activity_FedAvgLoss_confidence_interval, color='orange', alpha=0.2)
ax5.fill_between(rounds, data_rmse_age_FedFairLoss_means - data_rmse_age_FedFairLoss_confidence_interval, data_rmse_age_FedFairLoss_means + data_rmse_age_FedFairLoss_confidence_interval, color='g', alpha=0.2)

ax5.set_xlabel("Round")
ax5.set_ylabel(r"$RMSE$", fontsize=14)
ax5.legend()


# Subplot 6
ax6.plot(rounds, data_rmse_activity_FedAvgExample_means, label=r"FedAvg($n$)", linestyle='-')
ax6.plot(rounds, data_rmse_activity_FedAvgLoss_means, label=r"FedAvg($\ell$)", linestyle='-')
ax6.plot(rounds, data_rmse_gender_FedFairLoss_means, label=r"FedFair($\ell$)", linestyle='-')

ax6.fill_between(rounds, data_rmse_activity_FedAvgExample_means - data_rmse_activity_FedAvgExample_confidence_interval, data_rmse_activity_FedAvgExample_means + data_rmse_activity_FedAvgExample_confidence_interval, color='b', alpha=0.2)
ax6.fill_between(rounds, data_rmse_activity_FedAvgLoss_means - data_rmse_activity_FedAvgLoss_confidence_interval, data_rmse_activity_FedAvgLoss_means + data_rmse_activity_FedAvgLoss_confidence_interval, color='orange', alpha=0.2)
ax6.fill_between(rounds, data_rmse_gender_FedFairLoss_means - data_rmse_gender_FedFairLoss_confidence_interval, data_rmse_gender_FedFairLoss_means + data_rmse_gender_FedFairLoss_confidence_interval, color='g', alpha=0.2)

ax6.set_xlabel("Round")
ax6.set_ylabel(r"$RMSE$", fontsize=14)
ax6.legend()

plt.subplots_adjust(hspace=0.8, wspace=0.5)

# Ajustar espaçamento entre subplots
plt.tight_layout()
plt.show()
