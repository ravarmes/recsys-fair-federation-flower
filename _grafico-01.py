import matplotlib.pyplot as plt

from _resultados_FedAvgExample import data_rgrp_activity_FedAvg
from _resultados_FedAvgExample import data_rgrp_age_FedAvg
from _resultados_FedAvgExample import data_rgrp_gender_FedAvg
from _resultados_FedAvgExample import data_rmse_FedAvg

from _resultados_FedAvgLoss import data_rgrp_activity_FedLoss
from _resultados_FedAvgLoss import data_rgrp_age_FedLoss
from _resultados_FedAvgLoss import data_rgrp_gender_FedLoss
from _resultados_FedAvgLoss import data_rmse_FedLoss

from _resultados_FedFair import data_rgrp_activity_FairFed
from _resultados_FedFair import data_rgrp_age_FairFed
from _resultados_FedFair import data_rgrp_gender_FairFed
from _resultados_FedFair import data_rmse_activity_FairFed
from _resultados_FedFair import data_rmse_age_FairFed
from _resultados_FedFair import data_rmse_gender_FairFed


import matplotlib.pyplot as plt

# Criação da figura e dos subplots
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 9))

# Subplot 1
ax1.plot(data_rgrp_activity_FedAvg["Round"], data_rgrp_activity_FedAvg["RgrpActivity"], label=r"FedAvg", linestyle='-')
ax1.plot(data_rgrp_activity_FedLoss["Round"], data_rgrp_activity_FedLoss["RgrpActivity"], label=r"Fed($\ell$)", linestyle='-')
ax1.plot(data_rgrp_activity_FairFed["Round"], data_rgrp_activity_FairFed["RgrpActivity"], label=r"FairFed($\ell$)", linestyle='-')

ax1.set_ylabel(r"$R_{grp}$", fontsize=14)
ax1.set_title(r"Atividade")
ax1.legend()

# Subplot 2
ax2.plot(data_rgrp_age_FedAvg["Round"], data_rgrp_age_FedAvg["RgrpAge"], label=r"FedAvg", linestyle='-')
ax2.plot(data_rgrp_age_FedLoss["Round"], data_rgrp_age_FedLoss["RgrpAge"], label=r"Fed($\ell$)", linestyle='-')
ax2.plot(data_rgrp_age_FairFed["Round"], data_rgrp_age_FairFed["RgrpAge"], label=r"FairFed($\ell$)", linestyle='-')

#ax2.set_ylabel(r"$R_{grp}$", fontsize=14)
ax2.set_title(r"Idade")
ax2.legend()


# Subplot 3
ax3.plot(data_rgrp_gender_FedAvg["Round"], data_rgrp_gender_FedAvg["RgrpGender"], label=r"FedAvg", linestyle='-')
ax3.plot(data_rgrp_gender_FedLoss["Round"], data_rgrp_gender_FedLoss["RgrpGender"], label=r"Fed($\ell$)", linestyle='-')
ax3.plot(data_rgrp_gender_FairFed["Round"], data_rgrp_gender_FairFed["RgrpGender"], label=r"FairFed($\ell$)", linestyle='-')

#ax3.set_ylabel(r"$R_{grp}$", fontsize=14)
ax3.set_title(r"Gênero")
ax3.legend()


# Subplot 4
ax4.plot(data_rmse_FedAvg["Round"], data_rmse_FedAvg["RMSE"], label=r"FedAvg", linestyle='-')
ax4.plot(data_rmse_FedLoss["Round"], data_rmse_FedLoss["RMSE"], label=r"Fed($\ell$)", linestyle='-')
ax4.plot(data_rmse_activity_FairFed["Round"], data_rmse_activity_FairFed["RMSE"], label=r"FairFed($\ell$)", linestyle='-')

ax4.set_xlabel("Round")
ax4.set_ylabel(r"$RMSE$", fontsize=14)
# ax4.set_title(r"Atividade")
ax4.legend()

# Subplot 5
ax5.plot(data_rmse_FedAvg["Round"], data_rmse_FedAvg["RMSE"], label=r"FedAvg", linestyle='-')
ax5.plot(data_rmse_FedLoss["Round"], data_rmse_FedLoss["RMSE"], label=r"Fed($\ell$)", linestyle='-')
ax5.plot(data_rmse_age_FairFed["Round"], data_rmse_age_FairFed["RMSE"], label=r"FairFed($\ell$)", linestyle='-')

ax5.set_xlabel("Round")
#ax5.set_ylabel(r"$R_{grp}$", fontsize=14)
# ax5.set_title(r"Idade")
ax5.legend()


# Subplot 6
ax6.plot(data_rmse_FedAvg["Round"], data_rmse_FedAvg["RMSE"], label=r"FedAvg", linestyle='-')
ax6.plot(data_rmse_FedLoss["Round"], data_rmse_FedLoss["RMSE"], label=r"Fed($\ell$)", linestyle='-')
ax6.plot(data_rmse_gender_FairFed["Round"], data_rmse_gender_FairFed["RMSE"], label=r"FairFed($\ell$)", linestyle='-')


ax6.set_xlabel("Round")
#ax6.set_ylabel(r"$R_{grp}$", fontsize=14)
# ax6.set_title(r"Gênero")
ax6.legend()

plt.subplots_adjust(hspace=0.8, wspace=0.5)

# Ajustar espaçamento entre subplots
plt.tight_layout()
plt.show()
