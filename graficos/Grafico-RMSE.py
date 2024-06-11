import matplotlib.pyplot as plt

data_rmse = {
    "Round": list(range(0, 25)),
    "FedAvg": [
    0.0, 0.0, 0.0, 0.0, 0.2, 0.7, 0.4, 0.6, 0.8, 0.4,
    0.5, 0.8, 0.6, 0.6, 0.6, 0.7, 0.9, 0.8, 0.9, 0.9,
    0.8, 0.7, 0.5, 0.8, 0.7],
    "FedLossIndv": [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.7, 0.7, 0.8, 0.9,
    0.5, 0.6, 0.5, 0.2, 0.8, 0.6, 0.8, 1.0, 0.7, 1.0,
    0.8, 1.0, 0.6, 1.0, 0.8],
    "FairFedLossGroupActivity": [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.6, 0.5, 0.5, 0.8,
    0.5, 0.3, 0.5, 0.3, 0.8, 0.6, 0.5, 0.5, 0.6, 0.9,
    1.0, 0.6, 0.6, 0.9, 0.8],
    "FairFedLossGroupAge": [
    0.0, 0.0, 0.0, 0.0, 0.1, 0.4, 0.4, 0.8, 0.7, 0.5,
    0.7, 0.6, 0.7, 0.9, 0.8, 0.8, 0.4, 0.9, 0.6, 0.9,
    0.9, 0.8, 0.7, 0.7, 0.7]
,
    "FairFedLossGroupGender":  [
    0.0, 0.0, 0.0, 0.0, 0.1, 0.5, 0.5, 0.8, 0.1, 0.7,
    0.8, 0.6, 0.3, 0.9, 0.2, 0.5, 0.6, 0.5, 0.3, 0.7,
    0.7, 0.7, 0.9, 0.9, 0.7]

}

data_recall = {
    "Round": list(range(0, 25)),
    "FedAvg": [
    0.0, 0.0, 0.0, 0.0, 0.2, 0.7, 0.4, 0.6, 0.8, 0.4,
    0.5, 0.8, 0.6, 0.6, 0.6, 0.7, 0.9, 0.8, 0.9, 0.9,
    0.8, 0.7, 0.5, 0.8, 0.7],
    "FedLossIndv": [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.7, 0.7, 0.8, 0.9,
    0.5, 0.6, 0.5, 0.2, 0.8, 0.6, 0.8, 1.0, 0.7, 1.0,
    0.8, 1.0, 0.6, 1.0, 0.8],
    "FairFedLossGroupActivity": [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.6, 0.5, 0.5, 0.8,
    0.5, 0.3, 0.5, 0.3, 0.8, 0.6, 0.5, 0.5, 0.6, 0.9,
    1.0, 0.6, 0.6, 0.9, 0.8],
    "FairFedLossGroupAge": [
    0.0, 0.0, 0.0, 0.0, 0.1, 0.4, 0.4, 0.8, 0.7, 0.5,
    0.7, 0.6, 0.7, 0.9, 0.8, 0.8, 0.4, 0.9, 0.6, 0.9,
    0.9, 0.8, 0.7, 0.7, 0.7]
,
    "FairFedLossGroupGender":  [
    0.0, 0.0, 0.0, 0.0, 0.1, 0.5, 0.5, 0.8, 0.1, 0.7,
    0.8, 0.6, 0.3, 0.9, 0.2, 0.5, 0.6, 0.5, 0.3, 0.7,
    0.7, 0.7, 0.9, 0.9, 0.7]

}

data_rmse = {
    "Round": list(range(0, 25)),
    "FedAvg": [1.1645201444625854, 1.0938423871994019, 1.1099498271942139, 1.0983784198760986, 1.0669233798980713, 1.030524730682373, 1.0076098442077637, 0.993777871131897, 0.9872936010360718, 0.9874579310417175, 0.9897322654724121, 0.9946486353874207, 1.0001062154769897, 1.0043343305587769, 1.0080845355987549, 1.008457064628601, 1.0085766315460205, 1.007082462310791, 1.0062311887741089, 1.0068706274032593, 1.0067862272262573, 1.0087419748306274, 1.0141522884368896, 1.0159159898757935, 1.0210365056991577],
    "FedLossIndv": [1.154534935951233, 1.1196184158325195, 1.1417529582977295, 1.138689398765564, 1.1156989336013794, 1.0689215660095215, 1.0394608974456787, 1.0189247131347656, 1.0124248266220093, 1.0143638849258423, 1.0209097862243652, 1.0292022228240967, 1.034234642982483, 1.0368708372116089, 1.0374324321746826, 1.0374528169631958, 1.0369927883148193, 1.040049433708191, 1.04430091381073, 1.041678786277771, 1.0445189476013184, 1.0479038953781128, 1.0504499673843384, 1.0595613718032837, 1.0426335334777832],
    "FairFedLossGroupActivity": [1.1433990001678467, 1.1342644691467285, 1.145445704460144, 1.1400234699249268, 1.1024187803268433, 1.0632625818252563, 1.0306404829025269, 1.0147682428359985, 1.0100842714309692, 1.0112699270248413, 1.014522671699524, 1.0162739753723145, 1.0182409286499023, 1.0212211608886719, 1.0217827558517456, 1.020150899887085, 1.0181851387023926, 1.016127586364746, 1.0152902603149414, 1.0133719444274902, 1.015693187713623, 1.0161654949188232, 1.0151255130767822, 1.0401588678359985, 1.0193432569503784],
    "FairFedLossGroupAge": [1.230683445930481, 1.191461443901062, 1.2214837074279785, 1.2053853273391724, 1.1799521446228027, 1.1271016597747803, 1.0815538167953491, 1.0606739521026611, 1.0542610883712769, 1.0563547611236572, 1.05973482131958, 1.0622167587280273, 1.0698637962341309, 1.066306710243225, 1.0639629364013672, 1.0576655864715576, 1.0526093244552612, 1.0518447160720825, 1.0480000972747803, 1.0475820302963257, 1.041293740272522, 1.0377874374389648, 1.0793031454086304, 1.0368765592575073, 1.0276310443878174],
    "FairFedLossGroupGender":  [1.2513660192489624, 1.1453003883361816, 1.1669632196426392, 1.146763563156128, 1.107493281364441, 1.0625255107879639, 1.0340784788131714, 1.0163568258285522, 1.0092322826385498, 1.0010079145431519, 0.9964468479156494, 1.0065910816192627, 1.004364252090454, 1.0140713453292847, 1.0174165964126587, 1.0277018547058105, 1.0229432582855225, 1.0194227695465088, 1.014901041984558, 1.0131922960281372, 1.0103585720062256, 1.0070691108703613, 1.0084031820297241, 1.0078229904174805, 1.0067046880722046]
}

# --------------------------------------------------------------------------------------------


import matplotlib.pyplot as plt

# Criação da figura e dos subplots
fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(12, 9))

# Subplot 1
ax1.plot(data_rmse["Round"], data_rmse["FedAvg"], label=r"FedAvg", linestyle='-', linewidth=4, color = 'black')
ax1.plot(data_rmse["Round"], data_rmse["FedLossIndv"], label=r"Fed($\ell$)", linestyle='-', linewidth=4, color = 'gray')
ax1.plot(data_rmse["Round"], data_rmse["FairFedLossGroupActivity"], label=r"FairFed$(\lambda=0.2)$", linestyle='-', linewidth=4, color = 'blue')
ax1.set_xlabel("Round")
ax1.set_ylabel(r"$RMSE$", fontsize=14)
ax1.set_title(r"$RMSE$ Atividade")
ax1.legend()

# Subplot 2
ax2.plot(data_rmse["Round"], data_rmse["FedAvg"], label=r"FedAvg", linestyle='-', linewidth=4, color = 'black')
ax2.plot(data_rmse["Round"], data_rmse["FedLossIndv"], label=r"Fed($\ell$)", linestyle='-', linewidth=4, color = 'gray')
ax2.plot(data_rmse["Round"], data_rmse["FairFedLossGroupAge"], label=r"FairFed$(\lambda=0.2)$", linestyle='-', linewidth=4, color = 'blue')
ax2.set_xlabel("Round")
# ax2.set_ylabel(r"$RMSE$", fontsize=14)
ax2.set_title(r"$RMSE$ Idade")
ax2.legend()

# Subplot 3
ax3.plot(data_rmse["Round"], data_rmse["FedAvg"], label=r"FedAvg", linestyle='-', linewidth=4, color = 'black')
ax3.plot(data_rmse["Round"], data_rmse["FedLossIndv"], label=r"Fed($\ell$)", linestyle='-', linewidth=4, color = 'gray')
ax3.plot(data_rmse["Round"], data_rmse["FairFedLossGroupGender"], label=r"FairFed$(\lambda=0.6)$", linestyle='-', linewidth=4, color = 'green')
ax3.set_xlabel("Round")
# ax3.set_ylabel(r"$RMSE$", fontsize=14)
ax3.set_title(r"$RMSE$ Gênero")
ax3.legend()

# # Subplot 2
# ax2.plot(data_rmse["Round"], data_rmse["FedAvg"], label=r"FedAvg", linestyle='-', color = 'black')
# ax2.plot(data_rmse["Round"], data_rmse["FedLossIndv"], label=r"Fed($\ell$)", linestyle='-', color = 'gray')
# ax2.plot(data_rmse["Round"], data_rmse["FairFedLossGroupActivity"], label=r"FairFed$(\lambda=0.2)$", linestyle='-', color = 'blue')
# ax2.plot(data_rmse["Round"], data_rmse["FairFedLossGroupAge"], label=r"FairFed$(\lambda=0.2)$", linestyle='-', color = 'green')
# ax2.plot(data_rmse["Round"], data_rmse["FairFedLossGroupGender"], label=r"FairFed$(\lambda=0.6)$", linestyle='-', color = 'red')

# ax2.set_xlabel("Round")
# # ax1.set_ylabel(r"FairFed$(\lambda\ell)$", fontsize=14)

# ax2.set_title(r"$RMSE$")
# ax2.legend()



plt.subplots_adjust(hspace=0.8, wspace=0.5)

# Ajustar espaçamento entre subplots
plt.tight_layout()
plt.show()
