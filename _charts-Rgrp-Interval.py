import numpy as np
import matplotlib.pyplot as plt

from _results_MovieLens_Rgrp_26e14 import data_rgrp_activity_FedAvgExample_means, data_rgrp_activity_FedAvgExample_confidence_interval
from _results_MovieLens_Rgrp_26e14 import data_rgrp_activity_FedAvgLoss_means, data_rgrp_activity_FedAvgLoss_confidence_interval
from _results_MovieLens_Rgrp_26e14 import data_rgrp_activity_FedFairLoss_means, data_rgrp_activity_FedFairLoss_confidence_interval

from _results_MovieLens_Rgrp_26e14 import data_rgrp_age_FedAvgExample_means, data_rgrp_age_FedAvgExample_confidence_interval
from _results_MovieLens_Rgrp_26e14 import data_rgrp_age_FedAvgLoss_means, data_rgrp_age_FedAvgLoss_confidence_interval
from _results_MovieLens_Rgrp_26e14 import data_rgrp_age_FedFairLoss_means, data_rgrp_age_FedFairLoss_confidence_interval

from _results_MovieLens_Rgrp_26e14 import data_rgrp_gender_FedAvgExample_means, data_rgrp_gender_FedAvgExample_confidence_interval
from _results_MovieLens_Rgrp_26e14 import data_rgrp_gender_FedAvgLoss_means, data_rgrp_gender_FedAvgLoss_confidence_interval
from _results_MovieLens_Rgrp_26e14 import data_rgrp_gender_FedFairLoss_means, data_rgrp_gender_FedFairLoss_confidence_interval

from _results_MovieLens_RMSE_26e14 import data_rmse_activity_FedAvgExample_means, data_rmse_activity_FedAvgExample_confidence_interval
from _results_MovieLens_RMSE_26e14 import data_rmse_activity_FedAvgLoss_means, data_rmse_activity_FedAvgLoss_confidence_interval
from _results_MovieLens_RMSE_26e14 import data_rmse_activity_FedFairLoss_means, data_rmse_activity_FedFairLoss_confidence_interval
from _results_MovieLens_RMSE_26e14 import data_rmse_age_FedFairLoss_means, data_rmse_age_FedFairLoss_confidence_interval
from _results_MovieLens_RMSE_26e14 import data_rmse_gender_FedFairLoss_means, data_rmse_gender_FedFairLoss_confidence_interval

from _results_GoodBooks_Rgrp_05e02 import data_rgrp_activity_FedAvgExample_means_goodbooks, data_rgrp_activity_FedAvgExample_confidence_interval_goodbooks
from _results_GoodBooks_Rgrp_05e02 import data_rgrp_activity_FedAvgLoss_means_goodbooks, data_rgrp_activity_FedAvgLoss_confidence_interval_goodbooks
from _results_GoodBooks_Rgrp_05e02 import data_rgrp_activity_FedFairLoss_means_goodbooks, data_rgrp_activity_FedFairLoss_confidence_interval_goodbooks

from _results_GoodBooks_RMSE_05e02 import data_rmse_activity_FedAvgExample_means_goodbooks, data_rmse_activity_FedAvgExample_confidence_interval_goodbooks
from _results_GoodBooks_RMSE_05e02 import data_rmse_activity_FedAvgLoss_means_goodbooks, data_rmse_activity_FedAvgLoss_confidence_interval_goodbooks
from _results_GoodBooks_RMSE_05e02 import data_rmse_activity_FedFairLoss_means_goodbooks, data_rmse_activity_FedFairLoss_confidence_interval_goodbooks

rounds = list(range(0, 25))


# Criação da figura e dos subplots
fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(12, 9))

# Subplot 1
ax1.plot(rounds, data_rgrp_activity_FedAvgExample_means, label=r"FedAvg($n$)", linestyle='-')
ax1.plot(rounds, data_rgrp_activity_FedAvgLoss_means, label=r"FedAvg($\ell$)", linestyle='-')
ax1.plot(rounds, data_rgrp_activity_FedFairLoss_means, label=r"FedFair($\ell$)", linestyle='-')

ax1.fill_between(rounds, data_rgrp_activity_FedAvgExample_means - data_rgrp_activity_FedAvgExample_confidence_interval, data_rgrp_activity_FedAvgExample_means + data_rgrp_activity_FedAvgExample_confidence_interval, color='b', alpha=0.2)
ax1.fill_between(rounds, data_rgrp_activity_FedAvgLoss_means - data_rgrp_activity_FedAvgLoss_confidence_interval, data_rgrp_activity_FedAvgLoss_means + data_rgrp_activity_FedAvgLoss_confidence_interval, color='orange', alpha=0.2)
ax1.fill_between(rounds, data_rgrp_activity_FedFairLoss_means - data_rgrp_activity_FedFairLoss_confidence_interval, data_rgrp_activity_FedFairLoss_means + data_rgrp_activity_FedFairLoss_confidence_interval, color='g', alpha=0.2)

ax1.set_ylabel(r"$R_{grp}$", fontsize=14)
ax1.set_title(r"MovieLens - Activity")
ax1.legend(loc='lower center', bbox_to_anchor=(0.75, 0), bbox_transform=ax1.transAxes)

# Subplot 2
ax2.plot(rounds, data_rgrp_age_FedAvgExample_means, label=r"FedAvg($n$)", linestyle='-')
ax2.plot(rounds, data_rgrp_age_FedAvgLoss_means, label=r"FedAvg($\ell$)", linestyle='-')
ax2.plot(rounds, data_rgrp_age_FedFairLoss_means, label=r"FedFair($\ell$)", linestyle='-')

ax2.fill_between(rounds, data_rgrp_age_FedAvgExample_means - data_rgrp_age_FedAvgExample_confidence_interval, data_rgrp_age_FedAvgExample_means + data_rgrp_age_FedAvgExample_confidence_interval, color='b', alpha=0.2)
ax2.fill_between(rounds, data_rgrp_age_FedAvgLoss_means - data_rgrp_age_FedAvgLoss_confidence_interval, data_rgrp_age_FedAvgLoss_means + data_rgrp_age_FedAvgLoss_confidence_interval, color='orange', alpha=0.2)
ax2.fill_between(rounds, data_rgrp_age_FedFairLoss_means - data_rgrp_age_FedFairLoss_confidence_interval, data_rgrp_age_FedFairLoss_means + data_rgrp_age_FedFairLoss_confidence_interval, color='g', alpha=0.2)

ax2.set_title(r"MovieLens - Age")
ax2.legend(loc='lower center', bbox_to_anchor=(0.75, 0), bbox_transform=ax2.transAxes)


# Subplot 3
ax3.plot(rounds, data_rgrp_gender_FedAvgExample_means, label=r"FedAvg($n$)", linestyle='-')
ax3.plot(rounds, data_rgrp_gender_FedAvgLoss_means, label=r"FedAvg($\ell$)", linestyle='-')
ax3.plot(rounds, data_rgrp_gender_FedFairLoss_means, label=r"FedFair($\ell$)", linestyle='-')

ax3.fill_between(rounds, data_rgrp_gender_FedAvgExample_means - data_rgrp_gender_FedAvgExample_confidence_interval, data_rgrp_gender_FedAvgExample_means + data_rgrp_gender_FedAvgExample_confidence_interval, color='b', alpha=0.2)
ax3.fill_between(rounds, data_rgrp_gender_FedAvgLoss_means - data_rgrp_gender_FedAvgLoss_confidence_interval, data_rgrp_gender_FedAvgLoss_means + data_rgrp_gender_FedAvgLoss_confidence_interval, color='orange', alpha=0.2)
ax3.fill_between(rounds, data_rgrp_gender_FedFairLoss_means - data_rgrp_gender_FedFairLoss_confidence_interval, data_rgrp_gender_FedFairLoss_means + data_rgrp_gender_FedFairLoss_confidence_interval, color='g', alpha=0.2)

ax3.set_title(r"MovieLens - Gender")
ax3.legend(loc='lower center', bbox_to_anchor=(0.75, 0), bbox_transform=ax3.transAxes)

# Subplot 4
ax4.plot(rounds, data_rgrp_activity_FedAvgExample_means_goodbooks, label=r"FedAvg($n$)", linestyle='-')
ax4.plot(rounds, data_rgrp_activity_FedAvgLoss_means_goodbooks, label=r"FedAvg($\ell$)", linestyle='-')
ax4.plot(rounds, data_rgrp_activity_FedFairLoss_means_goodbooks, label=r"FedFair($\ell$)", linestyle='-')

ax4.fill_between(rounds, data_rgrp_activity_FedAvgExample_means_goodbooks - data_rgrp_activity_FedAvgExample_confidence_interval_goodbooks, data_rgrp_activity_FedAvgExample_means_goodbooks + data_rgrp_activity_FedAvgExample_confidence_interval_goodbooks, color='b', alpha=0.2)
ax4.fill_between(rounds, data_rgrp_activity_FedAvgLoss_means_goodbooks - data_rgrp_activity_FedAvgLoss_confidence_interval_goodbooks, data_rgrp_activity_FedAvgLoss_means_goodbooks + data_rgrp_activity_FedAvgLoss_confidence_interval_goodbooks, color='orange', alpha=0.2)
ax4.fill_between(rounds, data_rgrp_activity_FedFairLoss_means_goodbooks - data_rgrp_activity_FedFairLoss_confidence_interval_goodbooks, data_rgrp_activity_FedFairLoss_means_goodbooks + data_rgrp_activity_FedFairLoss_confidence_interval_goodbooks, color='g', alpha=0.2)

ax4.set_title(r"GoodBooks - Activity")
ax4.legend(loc='upper right', bbox_transform=ax1.transAxes)


# Subplot 5
ax5.plot(rounds, data_rmse_activity_FedAvgExample_means, label=r"FedAvg($n$)", linestyle='-')
ax5.plot(rounds, data_rmse_activity_FedAvgLoss_means, label=r"FedAvg($\ell$)", linestyle='-')
ax5.plot(rounds, data_rmse_activity_FedFairLoss_means, label=r"FedFair($\ell$)", linestyle='-')

ax5.fill_between(rounds, data_rmse_activity_FedAvgExample_means - data_rmse_activity_FedAvgExample_confidence_interval, data_rmse_activity_FedAvgExample_means + data_rmse_activity_FedAvgExample_confidence_interval, color='b', alpha=0.2)
ax5.fill_between(rounds, data_rmse_activity_FedAvgLoss_means - data_rmse_activity_FedAvgLoss_confidence_interval, data_rmse_activity_FedAvgLoss_means + data_rmse_activity_FedAvgLoss_confidence_interval, color='orange', alpha=0.2)
ax5.fill_between(rounds, data_rmse_activity_FedFairLoss_means - data_rmse_activity_FedFairLoss_confidence_interval, data_rmse_activity_FedFairLoss_means + data_rmse_activity_FedFairLoss_confidence_interval, color='g', alpha=0.2)

ax5.set_xlabel("Round")
ax5.set_ylabel(r"$RMSE$", fontsize=14)
ax5.legend()


# Subplot 6
ax6.plot(rounds, data_rmse_activity_FedAvgExample_means, label=r"FedAvg($n$)", linestyle='-')
ax6.plot(rounds, data_rmse_activity_FedAvgLoss_means, label=r"FedAvg($\ell$)", linestyle='-')
ax6.plot(rounds, data_rmse_age_FedFairLoss_means, label=r"FedFair($\ell$)", linestyle='-')

ax6.fill_between(rounds, data_rmse_activity_FedAvgExample_means - data_rmse_activity_FedAvgExample_confidence_interval, data_rmse_activity_FedAvgExample_means + data_rmse_activity_FedAvgExample_confidence_interval, color='b', alpha=0.2)
ax6.fill_between(rounds, data_rmse_activity_FedAvgLoss_means - data_rmse_activity_FedAvgLoss_confidence_interval, data_rmse_activity_FedAvgLoss_means + data_rmse_activity_FedAvgLoss_confidence_interval, color='orange', alpha=0.2)
ax6.fill_between(rounds, data_rmse_age_FedFairLoss_means - data_rmse_age_FedFairLoss_confidence_interval, data_rmse_age_FedFairLoss_means + data_rmse_age_FedFairLoss_confidence_interval, color='g', alpha=0.2)

ax6.set_xlabel("Round")
ax6.legend()


# Subplot 7
ax7.plot(rounds, data_rmse_activity_FedAvgExample_means, label=r"FedAvg($n$)", linestyle='-')
ax7.plot(rounds, data_rmse_activity_FedAvgLoss_means, label=r"FedAvg($\ell$)", linestyle='-')
ax7.plot(rounds, data_rmse_gender_FedFairLoss_means, label=r"FedFair($\ell$)", linestyle='-')

ax7.fill_between(rounds, data_rmse_activity_FedAvgExample_means - data_rmse_activity_FedAvgExample_confidence_interval, data_rmse_activity_FedAvgExample_means + data_rmse_activity_FedAvgExample_confidence_interval, color='b', alpha=0.2)
ax7.fill_between(rounds, data_rmse_activity_FedAvgLoss_means - data_rmse_activity_FedAvgLoss_confidence_interval, data_rmse_activity_FedAvgLoss_means + data_rmse_activity_FedAvgLoss_confidence_interval, color='orange', alpha=0.2)
ax7.fill_between(rounds, data_rmse_gender_FedFairLoss_means - data_rmse_gender_FedFairLoss_confidence_interval, data_rmse_gender_FedFairLoss_means + data_rmse_gender_FedFairLoss_confidence_interval, color='g', alpha=0.2)

ax7.set_xlabel("Round")
ax7.legend()


# Subplot 8
ax8.plot(rounds, data_rmse_activity_FedAvgExample_means_goodbooks, label=r"FedAvg($n$)", linestyle='-')
ax8.plot(rounds, data_rmse_activity_FedAvgLoss_means_goodbooks, label=r"FedAvg($\ell$)", linestyle='-')
ax8.plot(rounds, data_rmse_activity_FedFairLoss_means_goodbooks, label=r"FedFair($\ell$)", linestyle='-')

ax8.fill_between(rounds, data_rmse_activity_FedAvgExample_means_goodbooks - data_rmse_activity_FedAvgExample_confidence_interval_goodbooks, data_rmse_activity_FedAvgExample_means_goodbooks + data_rmse_activity_FedAvgExample_confidence_interval_goodbooks, color='b', alpha=0.2)
ax8.fill_between(rounds, data_rmse_activity_FedAvgLoss_means_goodbooks - data_rmse_activity_FedAvgLoss_confidence_interval_goodbooks, data_rmse_activity_FedAvgLoss_means_goodbooks + data_rmse_activity_FedAvgLoss_confidence_interval_goodbooks, color='orange', alpha=0.2)
ax8.fill_between(rounds, data_rmse_activity_FedFairLoss_means_goodbooks - data_rmse_activity_FedFairLoss_confidence_interval_goodbooks, data_rmse_activity_FedFairLoss_means_goodbooks + data_rmse_activity_FedFairLoss_confidence_interval_goodbooks, color='g', alpha=0.2)

ax8.set_xlabel("Round")
ax8.legend()

plt.subplots_adjust(hspace=0.2, wspace=0.05)

# Ajustar espaçamento entre subplots
plt.tight_layout()
plt.show()
